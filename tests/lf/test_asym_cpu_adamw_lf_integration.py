from __future__ import annotations

import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[2]
LF_SRC = ROOT.parent / "LlamaFactory" / "src"
if str(LF_SRC) not in sys.path:
    sys.path.insert(0, str(LF_SRC))

from asym_gemm.training.cpu_adam import AsymCPUAdamW
from asym_gemm.training.host_weight import HostWeight
from asym_gemm.training.offload import AsymFrozenEmbedding, AsymFrozenLayerNorm, AsymFrozenRMSNorm

try:
    import peft  # noqa: F401
    import llamafactory.model.adapter  # noqa: F401

    HAS_LF_ADAPTER_DEPS = True
except Exception:
    HAS_LF_ADAPTER_DEPS = False

try:
    import peft  # noqa: F401
    from llamafactory.train import trainer_utils
    from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer

    HAS_LF_RUNTIME_DEPS = True
except Exception:
    trainer_utils = None
    CustomSeq2SeqTrainer = None
    HAS_LF_RUNTIME_DEPS = False


requires_lf_runtime = pytest.mark.skipif(
    not HAS_LF_RUNTIME_DEPS,
    reason="LlamaFactory runtime dependencies are unavailable in this Python environment",
)
requires_lf_adapter = pytest.mark.skipif(
    not HAS_LF_ADAPTER_DEPS,
    reason="LlamaFactory adapter dependencies are unavailable in this Python environment",
)


def _cuda_available() -> bool:
    return torch.cuda.is_available() and torch.cuda.device_count() > 0


def _device() -> torch.device:
    return torch.device("cuda:0" if _cuda_available() else "cpu")


def _training_args(**overrides):
    values = {
        "learning_rate": 0.003,
        "adam_beta1": 0.8,
        "adam_beta2": 0.95,
        "adam_epsilon": 1e-6,
        "weight_decay": 0.07,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _finetuning_args(**overrides):
    values = {
        "use_asym_cpu_adamw": True,
        "asym_cpu_adamw_backend": "torch",
        "asym_cpu_adamw_pin_memory": False,
        "asym_cpu_adamw_fp32_master": True,
        "use_galore": False,
        "use_apollo": False,
        "loraplus_lr_ratio": None,
        "use_badam": False,
        "use_adam_mini": False,
        "use_muon": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class MixedLoRAModule(nn.Module):
    def __init__(self, *, device: torch.device) -> None:
        super().__init__()
        self.dense = nn.Module()
        self.dense.lora_A = nn.ModuleDict({"default": nn.Linear(4, 2, bias=False, device=device, dtype=torch.bfloat16)})
        self.dense.lora_B = nn.ModuleDict({"default": nn.Linear(2, 4, bias=False, device=device, dtype=torch.bfloat16)})
        self.gate_lora_A = nn.Parameter(torch.randn(2, 4, device=device, dtype=torch.bfloat16))
        self.gate_lora_B = nn.Parameter(torch.randn(4, 2, device=device, dtype=torch.bfloat16))
        self.up_lora_A = nn.Parameter(torch.randn(2, 4, device=device, dtype=torch.bfloat16))
        self.up_lora_B = nn.Parameter(torch.randn(4, 2, device=device, dtype=torch.bfloat16))
        self.down_lora_A = nn.Parameter(torch.randn(2, 4, device=device, dtype=torch.bfloat16))
        self.down_lora_B = nn.Parameter(torch.randn(4, 2, device=device, dtype=torch.bfloat16))


@requires_lf_adapter
def test_lf_lora_target_all_does_not_select_supported_moe_routers() -> None:
    from llamafactory.model.model_utils.misc import find_all_linear_modules
    from llamafactory.model.model_utils.visual import patch_target_modules
    from transformers.models.llama4.configuration_llama4 import Llama4TextConfig
    from transformers.models.llama4.modeling_llama4 import Llama4Router
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeConfig
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeTopKRouter
    from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeTopKRouter

    class TinyQwen3(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(model_type="qwen3_moe")
            cfg = Qwen3MoeConfig(hidden_size=8, num_experts=4, num_experts_per_tok=2, norm_topk_prob=True)
            self.mlp = nn.Module()
            self.mlp.gate = Qwen3MoeTopKRouter(cfg)
            self.mlp.gate_proj = nn.Linear(8, 8, bias=False)
            self.mlp.shared_expert_gate = nn.Linear(8, 1, bias=False)

    class TinyQwen35(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(model_type="qwen3_5_moe")
            cfg = Qwen3_5MoeConfig(hidden_size=8, num_experts=4, num_experts_per_tok=2)
            self.mlp = nn.Module()
            self.mlp.gate = Qwen3_5MoeTopKRouter(cfg)
            self.mlp.gate_proj = nn.Linear(8, 8, bias=False)
            self.mlp.shared_expert_gate = nn.Linear(8, 1, bias=False)

    class TinyLlama4(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(model_type="llama4_text")
            cfg = Llama4TextConfig(hidden_size=8, num_local_experts=4, num_experts_per_tok=2)
            self.feed_forward = nn.Module()
            self.feed_forward.router = Llama4Router(cfg)
            self.feed_forward.gate_proj = nn.Linear(8, 8, bias=False)
            self.feed_forward.down_proj = nn.Linear(8, 8, bias=False)

    class TinyCompositeLlama4(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(model_type="llama4")
            self.language_model = nn.Module()
            self.language_model.model = nn.Module()
            layer = nn.Module()
            layer.feed_forward = nn.Module()
            cfg = Llama4TextConfig(hidden_size=8, num_local_experts=4, num_experts_per_tok=2)
            layer.feed_forward.router = Llama4Router(cfg)
            layer.feed_forward.shared_expert = nn.Module()
            layer.feed_forward.shared_expert.gate_proj = nn.Linear(8, 8, bias=False)
            layer.feed_forward.shared_expert.down_proj = nn.Linear(8, 8, bias=False)
            self.language_model.model.layers = nn.ModuleList([layer])
            self.vision_model = nn.Module()
            self.multi_modal_projector = nn.Linear(8, 8, bias=False)
            self.lm_head = nn.Linear(8, 8, bias=False)

    qwen3_targets = set(find_all_linear_modules(TinyQwen3(), freeze_vision_tower=False))
    qwen35_targets = set(find_all_linear_modules(TinyQwen35(), freeze_vision_tower=False))
    llama4_targets = set(find_all_linear_modules(TinyLlama4(), freeze_vision_tower=False))
    composite_llama4 = TinyCompositeLlama4()
    composite_llama4_targets = set(find_all_linear_modules(composite_llama4, freeze_vision_tower=False))
    patched_composite_llama4_targets = set(
        patch_target_modules(
            composite_llama4,
            SimpleNamespace(
                freeze_vision_tower=False,
                freeze_multi_modal_projector=True,
                freeze_language_model=False,
            ),
            list(composite_llama4_targets),
        )
    )

    assert "gate" not in qwen3_targets
    assert {"gate_proj", "shared_expert_gate"} <= qwen3_targets
    assert "gate" not in qwen35_targets
    assert {"gate_proj", "shared_expert_gate"} <= qwen35_targets
    assert "router" not in llama4_targets
    assert {"gate_proj", "down_proj"} <= llama4_targets
    assert "router" not in composite_llama4_targets
    assert not any(target.endswith(".router") for target in patched_composite_llama4_targets)


@requires_lf_adapter
def test_lf_plain_lora_all_adapter_adds_qwen3_fused_expert_lora() -> None:
    from llamafactory.model.adapter import _setup_lora_tuning
    from llamafactory.model.model_utils.fused_moe_lora import count_fused_moe_lora
    from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM

    config = Qwen3MoeConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        moe_intermediate_size=8,
        num_experts=4,
        num_experts_per_tok=2,
        norm_topk_prob=True,
        output_router_logits=False,
        tie_word_embeddings=False,
    )
    model = Qwen3MoeForCausalLM(config)

    model_args = SimpleNamespace(
        adapter_name_or_path=None,
        adapter_folder=None,
        offload_folder=None,
        cache_dir=None,
        model_revision="main",
        hf_hub_token=None,
        use_kt=False,
        use_unsloth=False,
        use_asym_gemm=False,
        resize_vocab=False,
    )
    finetuning_args = SimpleNamespace(
        finetuning_type="lora",
        create_new_adapter=True,
        lora_target=["all"],
        freeze_vision_tower=False,
        freeze_multi_modal_projector=True,
        freeze_language_model=False,
        use_llama_pro=False,
        freeze_trainable_layers=0,
        use_dora=False,
        use_rslora=False,
        lora_rank=2,
        lora_alpha=4,
        lora_dropout=0.0,
        additional_target=None,
        pissa_init=False,
        pissa_iter=-1,
        oft_rank=0,
        oft_block_size=0,
        module_dropout=0.0,
    )

    model = _setup_lora_tuning(
        config,
        model,
        model_args,
        finetuning_args,
        is_trainable=True,
        cast_trainable_params_to_fp32=False,
    )
    trainable_names = [name for name, param in model.named_parameters() if param.requires_grad]

    assert count_fused_moe_lora(model) == {"modules": 1, "tensors": 6, "parameters": 576}
    assert any("mlp.experts._lf_fused_lora_params.gate_lora_a" in name for name in trainable_names)
    assert any("mlp.experts._lf_fused_lora_params.down_lora_b" in name for name in trainable_names)
    assert any("self_attn.q_proj.lora_A" in name for name in trainable_names)
    assert all(".mlp.gate." not in name and not name.endswith(".mlp.gate.weight") for name in trainable_names)


@requires_lf_adapter
def test_lf_peft_lora_all_does_not_add_adapter_to_qwen_moe_routers() -> None:
    from peft import LoraConfig, TaskType, get_peft_model
    from llamafactory.model.model_utils.fused_moe_lora import apply_fused_moe_lora, count_fused_moe_lora
    from llamafactory.model.model_utils.misc import find_all_linear_modules
    from llamafactory.model.model_utils.visual import patch_target_modules
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForCausalLM
    from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM

    def wrap_like_lf_lora_all(model: nn.Module) -> nn.Module:
        target_modules = find_all_linear_modules(model, freeze_vision_tower=False)
        target_modules = patch_target_modules(
            model,
            SimpleNamespace(
                freeze_vision_tower=False,
                freeze_multi_modal_projector=True,
                freeze_language_model=False,
            ),
            target_modules,
        )
        model = get_peft_model(
            model,
            LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=2,
                lora_alpha=4,
                lora_dropout=0.0,
                target_modules=target_modules,
            ),
        )
        apply_fused_moe_lora(model, rank=2, alpha=4)
        return model

    qwen3 = wrap_like_lf_lora_all(
        Qwen3MoeForCausalLM(
            Qwen3MoeConfig(
                vocab_size=64,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=4,
                num_key_value_heads=2,
                moe_intermediate_size=8,
                num_experts=4,
                num_experts_per_tok=2,
                norm_topk_prob=True,
                output_router_logits=False,
                tie_word_embeddings=False,
            )
        )
    )
    qwen35 = wrap_like_lf_lora_all(
        Qwen3_5MoeForCausalLM(
            Qwen3_5MoeTextConfig(
                vocab_size=64,
                hidden_size=16,
                num_hidden_layers=1,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=4,
                linear_key_head_dim=4,
                linear_value_head_dim=4,
                linear_num_key_heads=4,
                linear_num_value_heads=4,
                moe_intermediate_size=8,
                shared_expert_intermediate_size=8,
                num_experts=4,
                num_experts_per_tok=2,
                output_router_logits=False,
                tie_word_embeddings=False,
            )
        )
    )

    for model in (qwen3, qwen35):
        lora_module_names = [
            name
            for name, _module in model.named_modules()
            if "lora_A" in name or "lora_B" in name or "_lf_fused_lora_params" in name
        ]
        trainable_names = [name for name, param in model.named_parameters() if param.requires_grad]

        assert count_fused_moe_lora(model)["modules"] == 1
        assert all(".mlp.gate" not in name for name in lora_module_names)
        assert all(".mlp.gate" not in name for name in trainable_names)
        assert any("mlp.experts._lf_fused_lora_params.gate_lora" in name for name in trainable_names)

    assert any("mlp.shared_expert_gate.lora_A" in name for name, _param in qwen35.named_parameters())


@requires_lf_adapter
def test_lf_full_tuning_keeps_qwen_moe_routers_trainable() -> None:
    from llamafactory.model.adapter import _setup_full_tuning
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForCausalLM
    from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM

    def router_trainable_names(model: nn.Module) -> list[str]:
        return [name for name, param in model.named_parameters() if name.endswith(".mlp.gate.weight") and param.requires_grad]

    qwen3 = Qwen3MoeForCausalLM(
        Qwen3MoeConfig(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            moe_intermediate_size=8,
            num_experts=4,
            num_experts_per_tok=2,
            norm_topk_prob=True,
            output_router_logits=False,
            tie_word_embeddings=False,
        )
    )
    qwen35 = Qwen3_5MoeForCausalLM(
        Qwen3_5MoeTextConfig(
            vocab_size=64,
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=4,
            linear_key_head_dim=4,
            linear_value_head_dim=4,
            linear_num_key_heads=4,
            linear_num_value_heads=4,
            moe_intermediate_size=8,
            shared_expert_intermediate_size=8,
            num_experts=4,
            num_experts_per_tok=2,
            output_router_logits=False,
            tie_word_embeddings=False,
        )
    )
    args = SimpleNamespace(
        freeze_vision_tower=False,
        freeze_multi_modal_projector=True,
        freeze_language_model=False,
    )

    for model in (qwen3, qwen35):
        assert router_trainable_names(model) == ["model.layers.0.mlp.gate.weight"]
        _setup_full_tuning(model, args, is_trainable=True, cast_trainable_params_to_fp32=False)
        assert router_trainable_names(model) == ["model.layers.0.mlp.gate.weight"]


@requires_lf_adapter
def test_split_asym_peft_dense_targets_skips_router_targets_but_keeps_dense_lora_targets() -> None:
    from asym_gemm.integrations.lf import parse_lf_offload_modules
    from llamafactory.model.adapter import split_asym_peft_dense_targets

    class MixedRouterAndDenseTargets(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            layer = nn.Module()
            layer.mlp = nn.Module()
            layer.mlp.gate = nn.Linear(4, 4, bias=False)
            layer.mlp.gate_proj = nn.Linear(4, 4, bias=False)
            layer.feed_forward = nn.Module()
            layer.feed_forward.router = nn.Linear(4, 4, bias=False)
            layer.feed_forward.down_proj = nn.Linear(4, 4, bias=False)
            self.layers = nn.ModuleList([layer])

    selection = parse_lf_offload_modules("none")
    peft_targets, asym_targets = split_asym_peft_dense_targets(
        MixedRouterAndDenseTargets(),
        ["gate", "router", "gate_proj", "down_proj"],
        selection,
    )

    assert peft_targets == ["gate_proj", "down_proj"]
    assert asym_targets == []


@requires_lf_adapter
def test_split_asym_peft_dense_targets_expands_mixed_shared_expert_suffix() -> None:
    from asym_gemm.integrations.lf import parse_lf_offload_modules
    from llamafactory.model.adapter import split_asym_peft_dense_targets

    class MixedSharedExpertTargets(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            layer = nn.Module()
            layer.mlp = nn.Module()
            layer.mlp.gate_proj = nn.Linear(4, 4, bias=False)
            layer.mlp.shared_expert = nn.Module()
            layer.mlp.shared_expert.gate_proj = nn.Linear(4, 4, bias=False)
            self.layers = nn.ModuleList([layer])

    selection = parse_lf_offload_modules("shared_experts")
    peft_targets, asym_targets = split_asym_peft_dense_targets(
        MixedSharedExpertTargets(),
        ["gate_proj"],
        selection,
    )

    assert peft_targets == ["layers.0.mlp.gate_proj"]
    assert asym_targets == ["layers.0.mlp.shared_expert.gate_proj"]


@requires_lf_adapter
def test_split_asym_peft_dense_targets_keeps_nonselected_suffix_broad() -> None:
    from asym_gemm.integrations.lf import parse_lf_offload_modules
    from llamafactory.model.adapter import split_asym_peft_dense_targets

    class MixedAttentionTargets(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            layer = nn.Module()
            layer.self_attn = nn.Module()
            layer.self_attn.q_proj = nn.Linear(4, 4, bias=False)
            layer.self_attn.k_proj = nn.Linear(4, 4, bias=False)
            layer.mlp = nn.Module()
            layer.mlp.gate_proj = nn.Linear(4, 4, bias=False)
            self.layers = nn.ModuleList([layer])

    selection = parse_lf_offload_modules("q_proj")
    peft_targets, asym_targets = split_asym_peft_dense_targets(
        MixedAttentionTargets(),
        ["q_proj", "k_proj", "gate_proj"],
        selection,
    )

    assert peft_targets == ["k_proj", "gate_proj"]
    assert asym_targets == ["layers.0.self_attn.q_proj"]


@requires_lf_runtime
@pytest.mark.skipif(not _cuda_available(), reason="LF AsymCPUAdamW integration tests require CUDA")
def test_create_custom_optimizer_returns_asym_cpu_adamw_for_all_lora_name_forms() -> None:
    model = MixedLoRAModule(device=_device())

    optimizer = trainer_utils.create_custom_optimizer(model, _training_args(), _finetuning_args())

    assert isinstance(optimizer, AsymCPUAdamW)
    assert optimizer.backend == "torch"
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.003)
    assert optimizer.param_groups[0]["betas"] == (0.8, 0.95)
    assert optimizer.param_groups[0]["eps"] == pytest.approx(1e-6)
    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(0.07)
    assert set(optimizer.param_names) == {
        "gate_lora_A",
        "gate_lora_B",
        "up_lora_A",
        "up_lora_B",
        "down_lora_A",
        "down_lora_B",
        "dense.lora_A.default.weight",
        "dense.lora_B.default.weight",
    }


@requires_lf_runtime
@pytest.mark.skipif(not _cuda_available(), reason="LF AsymCPUAdamW integration tests require CUDA")
def test_create_custom_optimizer_rejects_trainable_non_lora_param() -> None:
    model = MixedLoRAModule(device=_device())
    model.extra_dense = nn.Linear(4, 4, bias=False, device=_device(), dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="trainable_non_lora"):
        trainer_utils.create_custom_optimizer(model, _training_args(), _finetuning_args())


@requires_lf_runtime
@pytest.mark.skipif(not _cuda_available(), reason="LF AsymCPUAdamW integration tests require CUDA")
def test_create_custom_optimizer_rejects_cpu_resident_trainable_lora_param() -> None:
    model = MixedLoRAModule(device=_device())
    model.cpu_lora_A = nn.Parameter(torch.randn(2, 4, dtype=torch.bfloat16))

    with pytest.raises(ValueError, match="Stage 7"):
        trainer_utils.create_custom_optimizer(model, _training_args(), _finetuning_args())


@requires_lf_runtime
@pytest.mark.skipif(not _cuda_available(), reason="LF AsymCPUAdamW integration tests require CUDA")
def test_create_custom_optimizer_ignores_frozen_base_offload_owners() -> None:
    class FrozenBasePlusLoRA(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed_tokens = AsymFrozenEmbedding(nn.Embedding(8, 4, dtype=torch.bfloat16))
            self.layer_norm = AsymFrozenLayerNorm(nn.LayerNorm(4, dtype=torch.bfloat16))
            rms = nn.Module()
            rms.weight = nn.Parameter(torch.ones(4, dtype=torch.bfloat16), requires_grad=False)
            rms.variance_epsilon = 1e-6
            self.rms_norm = AsymFrozenRMSNorm(rms)
            self.host = HostWeight(torch.randn(4, 4, dtype=torch.bfloat16), pin_memory=False)
            self.lora_A = nn.ModuleDict({"default": nn.Linear(4, 2, bias=False, device=_device(), dtype=torch.bfloat16)})

    optimizer = trainer_utils.create_custom_optimizer(FrozenBasePlusLoRA(), _training_args(), _finetuning_args())

    assert isinstance(optimizer, AsymCPUAdamW)
    assert optimizer.param_names == ["lora_A.default.weight"]


@requires_lf_runtime
@pytest.mark.skipif(not _cuda_available(), reason="LF AsymCPUAdamW integration tests require CUDA")
def test_first_step_post_prepare_device_check_keeps_cpu_masters_and_cuda_lora() -> None:
    model = MixedLoRAModule(device=_device())
    optimizer = trainer_utils.create_custom_optimizer(model, _training_args(), _finetuning_args())
    for group in optimizer.param_groups:
        for param in group["params"]:
            param.grad = torch.full_like(param, 0.125)

    optimizer.step()

    summary = optimizer.asym_cpu_adamw_summary()
    assert summary["all_masters_on_cpu"] is True
    assert summary["all_cuda_params_on_cuda"] is True
    assert summary["last_step_grad_param_count"] == len(optimizer.param_names)


@requires_lf_runtime
def test_custom_seq2seq_trainer_create_optimizer_delegates_before_hf_default() -> None:
    source = inspect.getsource(CustomSeq2SeqTrainer.create_optimizer)

    assert "create_custom_optimizer(self.model, self.args, self.finetuning_args)" in source
    assert source.index("create_custom_optimizer") < source.index("super().create_optimizer")


def test_asym_cpu_first_loader_source_order_moves_after_adapter_conversion() -> None:
    loader_path = LF_SRC / "llamafactory" / "model" / "loader.py"
    source = loader_path.read_text(encoding="utf-8")
    init_pos = source.index("model = init_adapter(config, model, model_args, finetuning_args, is_trainable)")
    move_pos = source.index("_move_asym_cpu_first_model_to_device(model)", init_pos)

    assert init_pos < move_pos
    assert "model.to(device)" in source
