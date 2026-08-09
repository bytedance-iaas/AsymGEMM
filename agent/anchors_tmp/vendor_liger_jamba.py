"""Vendor the loss-only Liger patch for Jamba into the container's liger install.

Mirrors the house glm4_moe/hunyuan mechanism (model_integration.md): copies the
generic lce_forward as liger_kernel/transformers/model/jamba.py and appends a
class-level apply_liger_kernel_to_jamba to monkey_patch.py. Idempotent.
"""
import inspect
import os

import liger_kernel.transformers.model.glm4_moe as g
import liger_kernel.transformers.monkey_patch as mp

base = os.path.dirname(g.__file__)
src = inspect.getsource(g)
jamba_path = os.path.join(base, "jamba.py")
if not os.path.exists(jamba_path):
    with open(jamba_path, "w") as f:
        f.write(
            "# Vendored for AI21-Jamba2-Mini (AsymGEMM model_integration.md #7):\n"
            "# loss-only fused-LCE forward — the generic lce_forward, byte-for-byte\n"
            "# the glm4_moe pattern.\n" + src
        )
    print("model/jamba.py written")
else:
    print("model/jamba.py already present")

APPLIER = '''

def apply_liger_kernel_to_jamba(
    rope: bool = False,
    cross_entropy: bool = False,
    fused_linear_cross_entropy: bool = True,
    rms_norm: bool = False,
    swiglu: bool = False,
    model: PreTrainedModel = None,
) -> None:
    """Loss-only Liger patch for Jamba (model_type=jamba, AI21-Jamba2-Mini):
    64k vocab -> materialized fp32 logits are the peak-HBM driver (AsymGEMM
    model_integration.md #7). Only fused_linear_cross_entropy is implemented;
    other kernel flags raise (same contract as the glm4_moe/hunyuan vendored
    appliers)."""
    assert not (cross_entropy and fused_linear_cross_entropy), (
        "cross_entropy and fused_linear_cross_entropy cannot both be True."
    )
    if rope or rms_norm or swiglu or cross_entropy:
        raise NotImplementedError("jamba Liger patch is loss-only (fused_linear_cross_entropy).")

    from transformers.models.jamba import modeling_jamba

    from liger_kernel.transformers.model.jamba import lce_forward as jamba_lce_forward

    if fused_linear_cross_entropy:
        if model is not None:
            model.forward = MethodType(jamba_lce_forward, model)
        else:
            modeling_jamba.JambaForCausalLM.forward = jamba_lce_forward
'''

mp_path = mp.__file__
with open(mp_path) as f:
    s = f.read()
if "apply_liger_kernel_to_jamba" not in s:
    with open(mp_path, "a") as f:
        f.write(APPLIER)
    print("monkey_patch: applier appended")
else:
    print("monkey_patch: already present")

import importlib

importlib.invalidate_caches()
importlib.reload(mp)
fn = getattr(mp, "apply_liger_kernel_to_jamba", None)
assert fn is not None, "applier missing after reload"
print("import OK:", fn.__name__)
