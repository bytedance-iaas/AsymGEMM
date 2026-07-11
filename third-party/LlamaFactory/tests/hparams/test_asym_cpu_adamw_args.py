from __future__ import annotations

from types import SimpleNamespace

import pytest
from transformers.training_args import ParallelMode

import llamafactory.hparams.parser as parser_module
from llamafactory.hparams import get_train_args


def _base_args(tmp_path):
    return {
        "model_name_or_path": "dummy",
        "stage": "sft",
        "do_train": True,
        "finetuning_type": "lora",
        "lora_rank": 8,
        "lora_dropout": 0.0,
        "lora_target": "all",
        "dataset": "dummy",
        "template": "qwen3",
        "cutoff_len": 128,
        "output_dir": str(tmp_path / "out"),
        "overwrite_output_dir": True,
        "max_steps": 1,
        "per_device_train_batch_size": 1,
        "max_grad_norm": 0.0,
        "report_to": "none",
        "save_strategy": "no",
        "use_asym_gemm": True,
        "asym_backend": "asym",
        "asym_precision": "bf16",
        "asym_offload_modules": "routed_experts",
        "asym_expert_recompute_policy": "none",
        "asym_router_mode": "whole",
        "pure_bf16": True,
        "use_asym_cpu_adamw": True,
        "asym_cpu_adamw_backend": "torch",
        "asym_cpu_adamw_fp32_master": True,
        "asym_cpu_adamw_grad_offload": False,
    }


@pytest.fixture(autouse=True)
def _stable_parser_environment(monkeypatch):
    monkeypatch.setattr(parser_module, "is_torch_bf16_gpu_available", lambda: True)
    monkeypatch.setattr(parser_module, "is_torch_npu_available", lambda: False)
    monkeypatch.setattr(parser_module, "is_deepspeed_zero3_enabled", lambda: False)


def test_asym_cpu_adamw_accepts_supported_lora_sft_args(tmp_path):
    model_args, _, training_args, finetuning_args, _ = get_train_args(_base_args(tmp_path))

    assert model_args.use_asym_gemm is True
    assert training_args.do_train is True
    assert finetuning_args.use_asym_cpu_adamw is True
    assert finetuning_args.asym_cpu_adamw_backend == "torch"
    assert finetuning_args.asym_cpu_adamw_fp32_master is True
    assert finetuning_args.asym_cpu_adamw_grad_offload is False


def test_asym_cpu_adamw_rejects_without_asym_gemm(tmp_path):
    args = _base_args(tmp_path)
    args["use_asym_gemm"] = False

    with pytest.raises(ValueError, match="requires `use_asym_gemm=true`"):
        get_train_args(args)


def test_asym_cpu_adamw_grad_offload_requires_cpuadamw(tmp_path):
    args = _base_args(tmp_path)
    args["use_asym_cpu_adamw"] = False
    args["asym_cpu_adamw_grad_offload"] = True

    with pytest.raises(ValueError, match="requires `use_asym_cpu_adamw=true`"):
        get_train_args(args)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("stage", "pt", "SFT stage|only supports SFT"),
        ("do_train", False, "requires `do_train=true`"),
        ("finetuning_type", "freeze", "LoRA finetuning|only supports LoRA"),
        ("pure_bf16", False, "pure_bf16=true"),
        ("asym_cpu_adamw_fp32_master", False, "fp32_master=true"),
        ("loraplus_lr_ratio", 2.0, "LoRA\\+"),
        ("optim", "sgd", "installed Transformers default"),
    ],
)
def test_asym_cpu_adamw_rejects_unsupported_train_args(tmp_path, field, value, message):
    args = _base_args(tmp_path)
    args[field] = value

    with pytest.raises(ValueError, match=message):
        get_train_args(args)


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("use_galore", "LoRA with GaLore"),
        ("use_apollo", "LoRA with GaLore"),
        ("use_badam", "LoRA with GaLore"),
        ("use_adam_mini", "use_adam_mini"),
        ("use_muon", "use_muon"),
        ("use_hyper_parallel", "use_hyper_parallel"),
    ],
)
def test_asym_cpu_adamw_rejects_other_custom_optimizer_flags(tmp_path, field, message):
    args = _base_args(tmp_path)
    args[field] = True

    with pytest.raises(ValueError, match=message):
        get_train_args(args)


def test_asym_cpu_adamw_rejects_use_mca_in_guard_directly():
    model_args = SimpleNamespace(use_asym_gemm=True)
    training_args = SimpleNamespace(
        do_train=True,
        deepspeed=None,
        parallel_mode=ParallelMode.NOT_PARALLEL,
        optim=parser_module._field_default(parser_module.TrainingArguments, "optim"),
        load_best_model_at_end=False,
        resume_from_checkpoint=None,
    )
    finetuning_args = SimpleNamespace(
        stage="sft",
        finetuning_type="lora",
        pure_bf16=True,
        asym_cpu_adamw_backend="torch",
        asym_cpu_adamw_fp32_master=True,
        asym_cpu_adamw_grad_offload=False,
        loraplus_lr_ratio=None,
        use_galore=False,
        use_apollo=False,
        use_badam=False,
        use_adam_mini=False,
        use_muon=False,
        use_mca=True,
        use_hyper_parallel=False,
    )

    with pytest.raises(ValueError, match="use_mca"):
        parser_module._verify_asym_cpu_adamw_args(model_args, training_args, finetuning_args)


def test_asym_cpu_adamw_rejects_deepspeed_or_zero(tmp_path, monkeypatch):
    args = _base_args(tmp_path)
    monkeypatch.setattr(parser_module, "is_deepspeed_zero3_enabled", lambda: True)

    with pytest.raises(ValueError, match="DeepSpeed|ZeRO"):
        get_train_args(args)


def test_asym_cpu_adamw_rejects_non_single_device_parallel_mode(tmp_path, monkeypatch):
    args = _base_args(tmp_path)
    monkeypatch.setattr(
        parser_module.TrainingArguments,
        "parallel_mode",
        property(lambda self: ParallelMode.NOT_DISTRIBUTED),
    )

    with pytest.raises(ValueError, match="single-process single-device"):
        get_train_args(args)


def test_asym_cpu_adamw_rejects_explicit_resume(tmp_path):
    args = _base_args(tmp_path)
    args["resume_from_checkpoint"] = str(tmp_path / "checkpoint-1")

    with pytest.raises(ValueError, match="checkpoint resume"):
        get_train_args(args)


def test_asym_cpu_adamw_rejects_output_dir_auto_resume(tmp_path):
    args = _base_args(tmp_path)
    checkpoint_dir = tmp_path / "out" / "checkpoint-1"
    checkpoint_dir.mkdir(parents=True)
    args["overwrite_output_dir"] = False

    with pytest.raises(ValueError, match="checkpoint resume"):
        get_train_args(args)


def test_asym_cpu_adamw_rejects_load_best_model_at_end(tmp_path):
    args = _base_args(tmp_path)
    args.update(
        {
            "load_best_model_at_end": True,
            "eval_strategy": "steps",
            "save_strategy": "steps",
            "eval_dataset": "dummy_eval",
        }
    )

    with pytest.raises(ValueError, match="load_best_model_at_end"):
        get_train_args(args)
