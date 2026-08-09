"""Diagnose the VLM mock dataset: do samples actually pad to seq_length?"""
import os
import sys

MODEL = "/scratch_local/user_data/shutian/kevin/cache/huggingface/hub/models--Qwen--Qwen3.5-35B-A3B/snapshots/59d61f3ce65a6d9863b86d2e96597125219dc754"

from dataclasses import dataclass

import torch  # noqa: F401
from megatron.bridge.data.vlm_datasets.mock_provider import MockVLMConversationProvider


@dataclass
class Ctx:
    train_samples: int = 8
    valid_samples: int = 0
    test_samples: int = 0


for seq in (8000, 32000):
    prov = MockVLMConversationProvider(
        seq_length=seq,
        hf_processor_path=MODEL,
        num_images=0,
        random_seed=42,
        pad_to_max_length=True,
        pad_to_multiple_of=128,
        enable_in_batch_packing=False,
        dataloader_type="single",
    )
    train_ds, _, _ = prov.build_datasets(Ctx())
    s0 = train_ds[0]
    print(f"== seq_length={seq}")
    for k, v in s0.items():
        try:
            print("   sample[0]", k, getattr(v, "shape", None) or (len(v) if hasattr(v, "__len__") else v))
        except Exception as e:
            print("   sample[0]", k, type(v), e)
    # collate a b=1 batch the way the loop does
    collate = getattr(train_ds, "collate_fn", None)
    if collate:
        b = collate([s0])
        for k, v in b.items():
            print("   batch", k, getattr(v, "shape", type(v)))
print("DIAG_DONE")
