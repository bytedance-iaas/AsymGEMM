# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
from dataclasses import dataclass, field
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch
from transformers import DataCollatorWithPadding

from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
from llamafactory.hparams import get_train_args
from llamafactory.model import load_model, load_tokenizer
from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer, _kt_clip_grad_norm_


DEMO_DATA = os.getenv("DEMO_DATA", "llamafactory/demo_data")

TINY_LLAMA3 = os.getenv("TINY_LLAMA3", "llamafactory/tiny-random-Llama-3")

TRAIN_ARGS = {
    "model_name_or_path": TINY_LLAMA3,
    "stage": "sft",
    "do_train": True,
    "finetuning_type": "lora",
    "dataset": "llamafactory/tiny-supervised-dataset",
    "dataset_dir": "ONLINE",
    "template": "llama3",
    "cutoff_len": 1024,
    "overwrite_output_dir": True,
    "per_device_train_batch_size": 1,
    "max_steps": 1,
    "report_to": "none",
}


def _install_fake_kt_sft(monkeypatch, **exports):
    kt_kernel = ModuleType("kt_kernel")
    kt_sft = ModuleType("kt_kernel.sft")
    for name, value in exports.items():
        setattr(kt_sft, name, value)
    kt_kernel.sft = kt_sft
    monkeypatch.setitem(sys.modules, "kt_kernel", kt_kernel)
    monkeypatch.setitem(sys.modules, "kt_kernel.sft", kt_sft)


def test_kt_clip_grad_norm_deduplicates_exact_grad_aliases_and_clips_cpu_views():
    shared_grad = torch.tensor([3.0, 4.0])
    param_a = torch.nn.Parameter(torch.zeros(2))
    param_b = torch.nn.Parameter(torch.zeros(2))
    param_a.grad = shared_grad
    param_b.grad = shared_grad

    view_backing_grad = torch.tensor([0.0, 12.0, 0.0, 0.0])
    param_c = torch.nn.Parameter(torch.zeros(2))
    param_d = torch.nn.Parameter(torch.zeros(2))
    param_c.grad = view_backing_grad[:2]
    param_d.grad = view_backing_grad[2:]

    total_norm, summary = _kt_clip_grad_norm_([param_a, param_b, param_c, param_d], max_norm=6.5)

    assert pytest.approx(float(total_norm), rel=1e-6) == 13.0
    assert summary["path"] == "kt_aware_dense"
    assert summary["operation"] == "clip"
    assert summary["unique_grad_tensors"] == 3
    assert summary["duplicate_grad_tensors"] == 1
    assert summary["cpu_grad_tensors"] == 3
    assert summary["clipped"] is True
    assert torch.allclose(shared_grad, torch.tensor([1.5, 2.0]))
    assert torch.allclose(view_backing_grad, torch.tensor([0.0, 6.0, 0.0, 0.0]))


def test_kt_clip_grad_norm_only_does_not_scale_gradients():
    grad = torch.tensor([3.0, 4.0])
    param = torch.nn.Parameter(torch.zeros(2))
    param.grad = grad

    total_norm, summary = _kt_clip_grad_norm_([param], max_norm=float("inf"))

    assert pytest.approx(float(total_norm), rel=1e-6) == 5.0
    assert summary["operation"] == "norm_only"
    assert summary["max_norm"] == "inf"
    assert summary["clipped"] is False
    assert torch.allclose(grad, torch.tensor([3.0, 4.0]))


def test_asym_sft_compute_loss_drops_training_logits():
    class DummyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))
            self.last_outputs = None

        def forward(self, input_ids, labels):
            logits = self.weight * torch.ones(
                int(input_ids.shape[0]),
                int(input_ids.shape[1]),
                4,
                device=input_ids.device,
            )
            outputs = {"loss": logits.sum(), "logits": logits}
            self.last_outputs = outputs
            return outputs

    trainer = object.__new__(CustomSeq2SeqTrainer)
    trainer.finetuning_args = SimpleNamespace(use_asft_loss=False)
    trainer.model_args = SimpleNamespace(asym_backend="asym")
    trainer.label_smoother = None
    model = DummyModel()
    inputs = {
        "input_ids": torch.ones(2, 3, dtype=torch.long),
        "labels": torch.ones(2, 3, dtype=torch.long),
    }

    loss = CustomSeq2SeqTrainer.compute_loss(trainer, model, inputs)

    assert torch.is_tensor(loss)
    assert model.last_outputs["logits"] is None


def test_custom_trainer_uses_kt_aware_clip_and_norm_for_arm_backend():
    class DummyAccelerator:
        def __init__(self) -> None:
            self.unscale_calls = 0

        def unscale_gradients(self) -> None:
            self.unscale_calls += 1

    model = torch.nn.Linear(2, 1, bias=False)
    model.weight.grad = torch.tensor([[3.0, 4.0]])
    trainer = object.__new__(CustomSeq2SeqTrainer)
    trainer.model_args = SimpleNamespace(use_kt=True, kt_backend="ARMBF16")
    trainer.args = SimpleNamespace(max_grad_norm=2.5)
    trainer.accelerator = DummyAccelerator()

    total_norm = CustomSeq2SeqTrainer._clip_grad_norm(trainer, model)

    assert pytest.approx(float(total_norm), rel=1e-6) == 5.0
    assert torch.allclose(model.weight.grad, torch.tensor([[1.5, 2.0]]))
    assert trainer.accelerator.unscale_calls == 1
    assert trainer._kt_grad_clip_last_summary["path"] == "kt_aware_dense"
    assert trainer._kt_grad_clip_last_summary["operation"] == "clip"

    norm = CustomSeq2SeqTrainer._get_grad_norm(trainer, model, grad_norm=None)

    assert pytest.approx(float(norm), rel=1e-6) == 2.5
    assert trainer.accelerator.unscale_calls == 2
    assert trainer._kt_grad_clip_last_summary["operation"] == "norm_only"


def test_custom_trainer_uses_asym_cpuadamw_offloaded_grad_clip_before_default() -> None:
    class DummyAccelerator:
        def __init__(self) -> None:
            self.unscale_calls = 0

        def unscale_gradients(self) -> None:
            self.unscale_calls += 1

    class DummyOptimizer:
        def __init__(self) -> None:
            self.calls: list[tuple[float, int]] = []

        def asym_cpu_adamw_grad_offload_enabled(self) -> bool:
            return True

        def asym_cpu_adamw_clip_grad_norm_(self, max_norm: float, *, chunk_elements: int):
            self.calls.append((max_norm, chunk_elements))
            summary = {
                "enabled": True,
                "path": "asym_cpu_adamw_grad_offload",
                "operation": "norm_only" if max_norm == float("inf") else "clip",
                "cpu_grad_numel": 12,
            }
            return torch.tensor(6.0), summary

    trainer = object.__new__(CustomSeq2SeqTrainer)
    trainer.model_args = SimpleNamespace(use_kt=False)
    trainer.args = SimpleNamespace(max_grad_norm=1.25)
    trainer.accelerator = DummyAccelerator()
    trainer.optimizer = DummyOptimizer()
    model = torch.nn.Linear(2, 1, bias=False)

    total_norm = CustomSeq2SeqTrainer._clip_grad_norm(trainer, model)

    assert pytest.approx(float(total_norm), rel=1e-6) == 6.0
    assert trainer.accelerator.unscale_calls == 1
    assert trainer.optimizer.calls == [(1.25, 8 * 1024 * 1024)]
    assert trainer._asym_cpu_adamw_grad_clip_last_summary["path"] == "asym_cpu_adamw_grad_offload"

    norm = CustomSeq2SeqTrainer._get_grad_norm(trainer, model, grad_norm=None)

    assert pytest.approx(float(norm), rel=1e-6) == 6.0
    assert trainer.accelerator.unscale_calls == 2
    assert trainer.optimizer.calls[-1] == (float("inf"), 8 * 1024 * 1024)
    assert trainer._asym_cpu_adamw_grad_clip_last_summary["operation"] == "norm_only"


def test_custom_trainer_create_optimizer_refreshes_kt_lora_pointers_once(monkeypatch):
    from llamafactory.train.sft import trainer as trainer_module

    calls: list[tuple[str, object, object | None]] = []

    class DummyOptimizer:
        def step(self, *args, **kwargs):
            calls.append(("step", args, kwargs))
            return "stepped"

    class DummyAccelerator:
        def __init__(self, unwrapped_model) -> None:
            self.unwrapped_model = unwrapped_model

        def unwrap_model(self, model):
            calls.append(("unwrap", model, None))
            return self.unwrapped_model

    def fake_super_create_optimizer(self):
        calls.append(("super_create_optimizer", self.optimizer, None))
        return self.optimizer

    def fake_update_kt_lora_pointers(model):
        calls.append(("kt_refresh", model, None))

    monkeypatch.setattr(trainer_module.Seq2SeqTrainer, "create_optimizer", fake_super_create_optimizer)
    _install_fake_kt_sft(monkeypatch, update_kt_lora_pointers=fake_update_kt_lora_pointers)

    wrapped_model = torch.nn.Linear(1, 1)
    unwrapped_model = torch.nn.Linear(1, 1)
    optimizer = DummyOptimizer()
    trainer = object.__new__(CustomSeq2SeqTrainer)
    trainer.model_args = SimpleNamespace(use_kt=True)
    trainer.optimizer = optimizer
    trainer.model = wrapped_model
    trainer.accelerator = DummyAccelerator(unwrapped_model)

    assert CustomSeq2SeqTrainer.create_optimizer(trainer) is optimizer
    wrapped_step = optimizer.step
    assert getattr(optimizer, "_kt_lora_pointer_refresh_wrapped") is True
    assert optimizer.step("arg", key="value") == "stepped"

    assert CustomSeq2SeqTrainer.create_optimizer(trainer) is optimizer
    assert optimizer.step is wrapped_step
    assert optimizer.step() == "stepped"

    assert calls == [
        ("super_create_optimizer", optimizer, None),
        ("step", ("arg",), {"key": "value"}),
        ("unwrap", wrapped_model, None),
        ("kt_refresh", unwrapped_model, None),
        ("super_create_optimizer", optimizer, None),
        ("step", (), {}),
        ("unwrap", wrapped_model, None),
        ("kt_refresh", unwrapped_model, None),
    ]


def test_custom_trainer_resume_loads_kt_sidecar_on_unwrapped_model(monkeypatch, tmp_path):
    from llamafactory.train.sft import trainer as trainer_module

    calls: list[tuple[str, object, object | None]] = []

    def fake_super_load(self, resume_from_checkpoint, model=None):
        calls.append(("super_load", resume_from_checkpoint, model))

    def fake_load_kt_moe_from_adapter(model, adapter_path):
        calls.append(("kt_load", model, adapter_path))

    class DummyAccelerator:
        def __init__(self, unwrapped_model) -> None:
            self.unwrapped_model = unwrapped_model

        def unwrap_model(self, model):
            calls.append(("unwrap", model, None))
            return self.unwrapped_model

    monkeypatch.setattr(trainer_module.Seq2SeqTrainer, "_load_from_checkpoint", fake_super_load)
    _install_fake_kt_sft(monkeypatch, load_kt_moe_from_adapter=fake_load_kt_moe_from_adapter)

    wrapped_model = torch.nn.Linear(1, 1)
    unwrapped_model = torch.nn.Linear(1, 1)
    trainer = object.__new__(CustomSeq2SeqTrainer)
    trainer.model_args = SimpleNamespace(use_kt=True)
    trainer.model = wrapped_model
    trainer.accelerator = DummyAccelerator(unwrapped_model)

    checkpoint = str(tmp_path / "checkpoint-1")
    CustomSeq2SeqTrainer._load_from_checkpoint(trainer, checkpoint)

    assert calls == [
        ("super_load", checkpoint, None),
        ("unwrap", wrapped_model, None),
        ("kt_load", unwrapped_model, checkpoint),
    ]


def test_custom_trainer_save_writes_kt_sidecar_after_base_save(monkeypatch, tmp_path):
    from llamafactory.train.sft import trainer as trainer_module

    calls: list[tuple[str, object, object | None]] = []

    def fake_super_save(self, output_dir=None, state_dict=None):
        calls.append(("super_save", output_dir, state_dict))

    def fake_save_kt_moe_to_adapter(model, output_dir):
        calls.append(("kt_save", model, output_dir))

    class DummyAccelerator:
        def __init__(self, unwrapped_model) -> None:
            self.unwrapped_model = unwrapped_model

        def unwrap_model(self, model):
            calls.append(("unwrap", model, None))
            return self.unwrapped_model

    monkeypatch.setattr(trainer_module.Seq2SeqTrainer, "_save", fake_super_save)
    _install_fake_kt_sft(monkeypatch, save_kt_moe_to_adapter=fake_save_kt_moe_to_adapter)

    wrapped_model = torch.nn.Linear(1, 1)
    unwrapped_model = torch.nn.Linear(1, 1)
    trainer = object.__new__(CustomSeq2SeqTrainer)
    trainer.model_args = SimpleNamespace(use_asym_gemm=False, use_kt=True)
    trainer.args = SimpleNamespace(output_dir=str(tmp_path / "default-output"))
    trainer.model = wrapped_model
    trainer.accelerator = DummyAccelerator(unwrapped_model)
    trainer.is_world_process_zero = lambda: True
    state_dict = {"weight": torch.ones(1)}
    output_dir = str(tmp_path / "checkpoint-1")

    CustomSeq2SeqTrainer._save(trainer, output_dir=output_dir, state_dict=state_dict)

    assert calls == [
        ("super_save", output_dir, state_dict),
        ("unwrap", wrapped_model, None),
        ("kt_save", unwrapped_model, output_dir),
    ]


@dataclass
class DataCollatorWithVerbose(DataCollatorWithPadding):
    verbose_list: list[dict[str, Any]] = field(default_factory=list)

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        features = [
            {k: v for k, v in feature.items() if k in ["input_ids", "attention_mask", "labels"]}
            for feature in features
        ]
        self.verbose_list.extend(features)
        batch = super().__call__(features)
        return {k: v[:, :1] for k, v in batch.items()}  # truncate input length


@pytest.mark.parametrize("disable_shuffling", [False, True])
def test_shuffle(disable_shuffling: bool):
    model_args, data_args, training_args, finetuning_args, _ = get_train_args(
        {
            "output_dir": os.path.join("output", f"shuffle{str(disable_shuffling).lower()}"),
            "disable_shuffling": disable_shuffling,
            **TRAIN_ARGS,
        }
    )
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    dataset_module = get_dataset(template, model_args, data_args, training_args, stage="sft", **tokenizer_module)
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)
    data_collator = DataCollatorWithVerbose(tokenizer=tokenizer)
    trainer = CustomSeq2SeqTrainer(
        model=model,
        args=training_args,
        finetuning_args=finetuning_args,
        data_collator=data_collator,
        **dataset_module,
        **tokenizer_module,
    )
    trainer.train()
    if disable_shuffling:
        assert data_collator.verbose_list[0]["input_ids"] == dataset_module["train_dataset"][0]["input_ids"]
    else:
        assert data_collator.verbose_list[0]["input_ids"] != dataset_module["train_dataset"][0]["input_ids"]
