from __future__ import annotations

import pytest
import torch
from torch import nn

from asym_gemm.training.decoder_activation_offload import (
    decoder_saved_tensor_offload_module_names,
    install_decoder_saved_tensor_offload,
    is_decoder_saved_tensor_offload_wrapper,
)


class SaveTensorModule(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _SaveForBackwardFunction.apply(x + 0).sum()


class _SaveForBackwardFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        return x * x

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor]:
        (x,) = ctx.saved_tensors
        return (grad_output * x * 2,)


def test_decoder_saved_tensor_offload_installs_in_place() -> None:
    module = SaveTensorModule()
    wrapper = install_decoder_saved_tensor_offload(module, min_bytes=0, require_grad=True)

    assert is_decoder_saved_tensor_offload_wrapper(module)
    assert install_decoder_saved_tensor_offload(module) is wrapper


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA saved tensor offload")
def test_decoder_saved_tensor_offload_preserves_backward_and_records_stats() -> None:
    torch.manual_seed(0)
    module = SaveTensorModule().cuda()
    install_decoder_saved_tensor_offload(module, min_bytes=0, require_grad=True)

    x = torch.randn(16, 16, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    expected_x = x.detach().clone().requires_grad_(True)
    expected = _SaveForBackwardFunction.apply(expected_x + 0).sum()
    expected.backward()

    loss = module(x)
    loss.backward()

    assert torch.allclose(x.grad, expected_x.grad, atol=0, rtol=0)
    stats = module._last_activation_offload_stats
    assert stats["decoder_saved_tensor_offload"] is True
    assert stats["num_offloads"] > 0
    assert stats["num_stages"] > 0
    assert stats["offloaded_bytes"] > 0
    assert stats["cpu_live_bytes"] == 0


def test_decoder_saved_tensor_offload_module_names() -> None:
    model = nn.Sequential(SaveTensorModule(), SaveTensorModule())
    install_decoder_saved_tensor_offload(model[0], min_bytes=0, require_grad=True)

    assert decoder_saved_tensor_offload_module_names(model) == ("0",)
