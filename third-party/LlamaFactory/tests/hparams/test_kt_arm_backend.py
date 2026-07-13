import os

import pytest

from llamafactory.hparams import get_train_args


def _base_args(kt_backend: str = "TORCHBF16") -> dict:
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
        "output_dir": "dummy_out",
        "overwrite_output_dir": True,
        "max_grad_norm": 0.0,
        "use_kt": True,
        "kt_backend": kt_backend,
        "kt_num_threads": 4,
        "kt_max_cache_depth": 2,
    }


@pytest.mark.parametrize("kt_backend", ["TORCHBF16", "ARMBF16", "ARMBF16_SFT", "KT_ARM"])
def test_local_kt_backend_uses_source_import_and_sets_env(kt_backend: str):
    model_args, _, training_args, _, _ = get_train_args(_base_args(kt_backend))

    assert model_args.kt_backend == kt_backend
    assert training_args.world_size == 1
    assert os.environ["ACCELERATE_KT_BACKEND"] == kt_backend
    assert os.environ["ACCELERATE_KT_NUM_THREADS"] == "4"
    assert os.environ["ACCELERATE_KT_MAX_CACHE_DEPTH"] == "2"


def test_unsupported_kt_backend_rejects_lora_dropout():
    args = _base_args("AMXBF16")
    args["lora_dropout"] = 0.1

    with pytest.raises(ValueError, match="lora_dropout"):
        get_train_args(args)


@pytest.mark.parametrize("kt_backend", ["ARMBF16", "ARMBF16_SFT", "KT_ARM"])
def test_arm_kt_backend_rejects_gradient_accumulation(kt_backend: str):
    args = _base_args(kt_backend)
    args["gradient_accumulation_steps"] = 2

    with pytest.raises(ValueError, match="gradient_accumulation_steps=1"):
        get_train_args(args)


@pytest.mark.parametrize("kt_backend", ["ARMBF16", "ARMBF16_SFT", "KT_ARM"])
def test_arm_kt_backend_allows_kt_aware_gradient_clipping(kt_backend: str):
    args = _base_args(kt_backend)
    args["max_grad_norm"] = 1.0

    model_args, _, training_args, _, _ = get_train_args(args)

    assert model_args.kt_backend == kt_backend
    assert training_args.max_grad_norm == 1.0


@pytest.mark.parametrize("kt_backend", ["ARMBF16", "ARMBF16_SFT", "KT_ARM"])
def test_arm_kt_backend_rejects_default_unvalidated_large_route_rank(monkeypatch, kt_backend: str):
    monkeypatch.delenv("KT_ARM_ALLOW_UNVALIDATED_ROUTE_RANK_WORK", raising=False)
    monkeypatch.delenv("KT_ARM_SFT_MAX_ROUTE_RANK_WORK", raising=False)
    monkeypatch.delenv("KT_ARM_SFT_TOKEN_CHUNK_SIZE", raising=False)
    monkeypatch.delenv("KT_ARM_SFT_TOP_K", raising=False)
    args = _base_args(kt_backend)
    args["cutoff_len"] = 7168
    args["per_device_train_batch_size"] = 4
    args["lora_rank"] = 64

    with pytest.raises(ValueError, match="route-rank work 14680064 exceeds"):
        get_train_args(args)


@pytest.mark.parametrize("kt_backend", ["ARMBF16", "ARMBF16_SFT", "KT_ARM"])
def test_arm_kt_backend_allows_large_route_rank_with_token_chunk(monkeypatch, kt_backend: str):
    monkeypatch.delenv("KT_ARM_ALLOW_UNVALIDATED_ROUTE_RANK_WORK", raising=False)
    monkeypatch.delenv("KT_ARM_SFT_MAX_ROUTE_RANK_WORK", raising=False)
    monkeypatch.setenv("KT_ARM_SFT_TOKEN_CHUNK_SIZE", "2048")
    monkeypatch.delenv("KT_ARM_SFT_TOP_K", raising=False)
    args = _base_args(kt_backend)
    args["cutoff_len"] = 7168
    args["per_device_train_batch_size"] = 4
    args["lora_rank"] = 64

    model_args, _, _, _, _ = get_train_args(args)

    assert model_args.kt_backend == kt_backend
    assert model_args.kt_model_max_length == 28672


@pytest.mark.parametrize("kt_backend", ["ARMBF16", "ARMBF16_SFT", "KT_ARM"])
def test_arm_kt_backend_honors_default_route_rank_limit_env(monkeypatch, kt_backend: str):
    monkeypatch.delenv("KT_ARM_ALLOW_UNVALIDATED_ROUTE_RANK_WORK", raising=False)
    monkeypatch.delenv("KT_ARM_SFT_MAX_ROUTE_RANK_WORK", raising=False)
    monkeypatch.delenv("KT_ARM_SFT_TOKEN_CHUNK_SIZE", raising=False)
    monkeypatch.setenv("KT_ARM_SFT_DEFAULT_MAX_ROUTE_RANK_WORK", "1024")
    args = _base_args(kt_backend)
    args["cutoff_len"] = 128
    args["per_device_train_batch_size"] = 2
    args["lora_rank"] = 8

    with pytest.raises(ValueError, match="KT_ARM_SFT_MAX_ROUTE_RANK_WORK=1024"):
        get_train_args(args)


@pytest.mark.parametrize("kt_backend", ["ARMBF16", "ARMBF16_SFT", "KT_ARM"])
def test_arm_kt_backend_rejects_invalid_token_chunk(monkeypatch, kt_backend: str):
    monkeypatch.setenv("KT_ARM_SFT_TOKEN_CHUNK_SIZE", "0")
    args = _base_args(kt_backend)

    with pytest.raises(ValueError, match="KT_ARM_SFT_TOKEN_CHUNK_SIZE"):
        get_train_args(args)


@pytest.mark.parametrize("kt_backend", ["ARMBF16", "ARMBF16_SFT", "KT_ARM"])
def test_arm_kt_backend_allows_explicit_unvalidated_large_route_rank(monkeypatch, kt_backend: str):
    monkeypatch.setenv("KT_ARM_ALLOW_UNVALIDATED_ROUTE_RANK_WORK", "1")
    monkeypatch.delenv("KT_ARM_SFT_MAX_ROUTE_RANK_WORK", raising=False)
    monkeypatch.delenv("KT_ARM_SFT_TOKEN_CHUNK_SIZE", raising=False)
    monkeypatch.delenv("KT_ARM_SFT_TOP_K", raising=False)
    args = _base_args(kt_backend)
    args["cutoff_len"] = 7168
    args["per_device_train_batch_size"] = 4
    args["lora_rank"] = 64

    model_args, _, _, _, _ = get_train_args(args)

    assert model_args.kt_backend == kt_backend


@pytest.mark.parametrize("kt_backend", ["ARMBF16", "ARMBF16_SFT", "KT_ARM"])
def test_arm_kt_backend_sets_default_backward_scratch_limit(monkeypatch, kt_backend: str):
    monkeypatch.delenv("KT_ARM_SFT_MAX_BACKWARD_SCRATCH_BYTES", raising=False)
    monkeypatch.delenv("KT_ARM_SFT_DEFAULT_MAX_BACKWARD_SCRATCH_BYTES", raising=False)
    monkeypatch.delenv("KT_ARM_ALLOW_UNVALIDATED_BACKWARD_SCRATCH", raising=False)
    args = _base_args(kt_backend)

    get_train_args(args)

    assert os.environ["KT_ARM_SFT_MAX_BACKWARD_SCRATCH_BYTES"] == "34359738368"


@pytest.mark.parametrize("kt_backend", ["ARMBF16", "ARMBF16_SFT", "KT_ARM"])
def test_arm_kt_backend_allows_unvalidated_backward_scratch_without_default(monkeypatch, kt_backend: str):
    monkeypatch.delenv("KT_ARM_SFT_MAX_BACKWARD_SCRATCH_BYTES", raising=False)
    monkeypatch.setenv("KT_ARM_ALLOW_UNVALIDATED_BACKWARD_SCRATCH", "1")
    args = _base_args(kt_backend)

    get_train_args(args)

    assert "KT_ARM_SFT_MAX_BACKWARD_SCRATCH_BYTES" not in os.environ


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("kt_tp_enabled", True, "kt_tp_enabled=true"),
        ("kt_threadpool_count", 2, "kt_threadpool_count"),
        ("kt_num_gpu_experts", 1, "GPU experts"),
    ],
)
@pytest.mark.parametrize("kt_backend", ["ARMBF16", "ARMBF16_SFT", "KT_ARM"])
def test_arm_kt_backend_rejects_unsupported_options(kt_backend: str, field: str, value, message: str):
    args = _base_args(kt_backend)
    args[field] = value

    with pytest.raises(ValueError, match=message):
        get_train_args(args)
