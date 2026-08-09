#!/usr/bin/env python3
"""NeMo / Megatron-Bridge LoRA SFT throughput runner (AsymGEMM baseline protocol).

Launched under torchrun by scripts/lf/profile_lora_nemo.sh, one process per
rank, inside the asym_sft container with .venv-nemo. Env-driven so the outer
driver mirrors the LF profiling drivers (WARMUP_STEPS/MAX_STEPS, seq|batch|ga,
LoRA r/alpha/dropout) while this file owns the Megatron-Bridge ConfigContainer.

Protocol parity with scripts/lf/profile_lora_lf_test_source.sh:
  - train_iters = WARMUP + MEASURED; per-iteration wall time is parsed by the
    driver from the "elapsed time per iteration (ms):" stdout record
    (log_interval=1); the first WARMUP iterations are excluded downstream.
  - mock GPT dataset pinned at the target seq length (the LF driver likewise
    trains on generated synthetic full-length rows; loss values are
    meaningless here, timings and memory are the measurement).
  - LoRA on every linear (attention + MLP + per-expert adapters), bf16.
  - Parallelism: TP=1 PP=1 CP=1, EP = all ranks ("normal EP"), the rest DP.

Memory-saving arms (NEMO_RECOMPUTE / NEMO_ACT_OFFLOAD):
  - recompute full: recompute_granularity=full, uniform, 1 (the strongest
    recompute Megatron offers; NeMo's analogue of the baselines' `recomp`).
  - act offload: fine_grained_activation_offloading + every offloadable
    module (Megatron cannot offload the full-recompute layer boundaries;
    cpu_offloading is mutually exclusive with recompute by upstream
    validation, so this is the strongest offload config it ships).
Model weights always stay HBM-resident (frozen bf16 shards) — Megatron has no
parameter/weight offload for frozen base weights; that is the point of this
baseline.
"""

import json
import os
import sys


def _env(name: str, default=None, cast=str):
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return cast(raw)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "")
    if raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def main() -> None:
    model_path = os.environ["NEMO_MODEL_PATH"]  # local HF snapshot dir
    seq_len = int(os.environ["NEMO_SEQ_LEN"])
    mbs = _env("NEMO_MICRO_BATCH", 1, int)
    ga = _env("NEMO_GRAD_ACCUM", 1, int)
    warmup = _env("NEMO_WARMUP_STEPS", 1, int)
    measured = _env("NEMO_MEASURED_STEPS", 2, int)
    ep = _env("NEMO_EP", None, int)
    tp = _env("NEMO_TP", 1, int)
    etp = _env("NEMO_ETP", 1, int)
    recompute = _env("NEMO_RECOMPUTE", "full")  # full | sel | none
    act_offload = _env_bool("NEMO_ACT_OFFLOAD", False)
    offload_modules = _env(
        "NEMO_OFFLOAD_MODULES",
        "core_attn,attn_proj,qkv_linear,attn_norm,mlp_norm,expert_fc1,moe_act",
    ).split(",")
    lora_rank = _env("NEMO_LORA_RANK", 64, int)
    lora_alpha = _env("NEMO_LORA_ALPHA", 16, int)
    lora_dropout = _env("NEMO_LORA_DROPOUT", 0.0, float)
    seed = _env("NEMO_SEED", 42, int)
    out_dir = _env("NEMO_OUT_DIR", os.path.join(os.getcwd(), "nemo_run_out"))
    load_weights = _env_bool("NEMO_LOAD_WEIGHTS", True)
    dispatcher = _env("NEMO_MOE_DISPATCHER", "alltoall")

    world = int(os.environ.get("WORLD_SIZE", "1"))
    if ep is None:
        ep = world
    dp = world // (tp * 1)
    gbs = mbs * dp * ga

    if act_offload:
        # mcore requires this with TE >= 2.10 so TE's offload hooks grab
        # activations only, never weights. Must be set before TE is imported.
        os.environ.setdefault("NVTE_CPU_OFFLOAD_V1", "1")

    import torch  # noqa: F401  (must precede TE so venv cublas is preloaded)
    from megatron.bridge import AutoBridge
    from megatron.bridge.peft.lora import LoRA
    from megatron.bridge.recipes.common import _peft_common, _peft_common_vlm
    from megatron.bridge.training.config import GPTDatasetConfig
    from megatron.bridge.training.finetune import finetune
    from megatron.bridge.training.mixed_precision import bf16_mixed

    # Family: Qwen3.5-35B-A3B ships as a ForConditionalGeneration (VL) HF
    # checkpoint; its bridge/model/data path differ from the text MoE models.
    with open(os.path.join(model_path, "config.json")) as fh:
        hf_cfg = json.load(fh)
    is_vlm = any("ConditionalGeneration" in a for a in hf_cfg.get("architectures", []))

    if is_vlm:
        import megatron.bridge.training.vlm_step as _vlm_step
        from megatron.bridge.training.vlm_step import forward_step as step_fn

        # Upstream bug guard: with data-parallel size 1 the collate receives
        # pad_to_max_length=False even though the provider is constructed with
        # True (verified via NEMO_DBG_PREP: cfg.dataset carries True, the
        # collate kwargs carry False; at dp=2 the flag arrives intact), which
        # silently trains ~128-token batches labelled as full-seq runs. Force
        # the flag at the collate boundary so every rank count pads to the
        # configured seq_length. NEMO_DEBUG_BATCH=1 prints the shapes that
        # actually reach the model for verification.
        import megatron.bridge.models.qwen_vl.data.collate_fn as _qcf

        _orig_prep_forced = _qcf.prepare_padded_or_packed_sequence_batch

        def _forced_prep(batch, **kw):
            if not kw.get("enable_in_batch_packing", False):
                kw["pad_to_max_length"] = True
            return _orig_prep_forced(batch, **kw)

        _qcf.prepare_padded_or_packed_sequence_batch = _forced_prep

        if _env_bool("NEMO_DEBUG_BATCH", False):
            _orig_get_batch = _vlm_step.get_batch

            def _dbg_get_batch(*a, **k):
                out = _orig_get_batch(*a, **k)
                try:
                    t = out[0]
                    lm = out[2]
                    print(
                        "NEMO_DBG_BATCH tokens="
                        f"{tuple(t.shape) if t is not None else None} "
                        f"lossmask_sum={float(lm.sum()) if lm is not None else None}",
                        flush=True,
                    )
                except Exception as exc:  # pragma: no cover
                    print("NEMO_DBG_BATCH err", repr(exc), flush=True)
                return out

            _vlm_step.get_batch = _dbg_get_batch

            _orig_prep_dbg = _qcf.prepare_padded_or_packed_sequence_batch

            def _dbg_prep(batch, **kw):
                print(
                    "NEMO_DBG_PREP "
                    + str({k: kw.get(k) for k in (
                        "sequence_length", "pad_to_max_length",
                        "pad_to_multiple_of", "enable_in_batch_packing")}),
                    flush=True,
                )
                return _orig_prep_dbg(batch, **kw)

            _qcf.prepare_padded_or_packed_sequence_batch = _dbg_prep

        cfg = _peft_common_vlm()
    else:
        from megatron.bridge.training.gpt_step import forward_step as step_fn

        cfg = _peft_common()

    bridge = AutoBridge.from_hf_pretrained(model_path, trust_remote_code=False)
    cfg.model = bridge.to_megatron_provider(load_weights=False)

    cfg.tokenizer.tokenizer_model = model_path

    # Parallelism: "normal EP" — experts sharded over all ranks, no TP/PP/CP.
    cfg.model.tensor_model_parallel_size = tp
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.pipeline_dtype = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = ep
    cfg.model.expert_tensor_parallel_size = etp
    cfg.model.sequence_parallel = tp > 1

    cfg.model.seq_length = seq_len

    # MoE plumbing: plain NCCL alltoall dispatcher (no DeepEP/HybridEP deps).
    cfg.model.moe_token_dispatcher_type = dispatcher
    cfg.model.moe_flex_dispatcher_backend = None
    cfg.model.moe_grouped_gemm = True
    cfg.model.moe_permute_fusion = True
    cfg.model.moe_shared_expert_overlap = False
    cfg.model.moe_router_force_load_balancing = False

    # Loss: fused cross entropy (NeMo's ligerloss analogue) — always on, like
    # every baseline cell (ligerloss1). The VL recipes ship impl "native".
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native" if is_vlm else "te"

    if is_vlm:
        # Text-only protocol parity: vision tower frozen and unused (no image
        # inputs); no MTP head (the LF baselines train the standard CE loss).
        cfg.model.freeze_vision_model = True
        cfg.model.freeze_vision_projection = True
        cfg.model.freeze_language_model = False

    # No MTP head for ANY model (glm4.5/qwen3.5 checkpoints ship one; the LF
    # baselines train the plain CE loss, and skipping it is generous to NeMo).
    if hasattr(cfg.model, "mtp_num_layers"):
        cfg.model.mtp_num_layers = None
    if hasattr(cfg.model, "mtp_loss_scaling_factor"):
        cfg.model.mtp_loss_scaling_factor = None

    cfg.model.attention_backend = None  # let TE pick (cuDNN fused on sm100)
    cfg.model.cuda_graph_impl = "none"
    # Needs apex's fused_weight_gradient_mlp_cuda, which the pip venv does not
    # ship; base weights are frozen under LoRA so the fusion buys nothing here.
    cfg.model.gradient_accumulation_fusion = False

    # Memory-saving arms.
    if recompute == "full":
        cfg.model.recompute_granularity = "full"
        cfg.model.recompute_method = "uniform"
        cfg.model.recompute_num_layers = 1
        cfg.model.recompute_modules = None
    elif recompute == "sel":
        cfg.model.recompute_granularity = "selective"
        cfg.model.recompute_method = None
        cfg.model.recompute_num_layers = None
        cfg.model.recompute_modules = _env(
            "NEMO_RECOMPUTE_MODULES", "core_attn,moe_act,layernorm,mlp"
        ).split(",")
    else:
        cfg.model.recompute_granularity = None
        cfg.model.recompute_method = None
        cfg.model.recompute_num_layers = None
        cfg.model.recompute_modules = None

    cfg.model.fine_grained_activation_offloading = act_offload
    if act_offload:
        if recompute == "full":
            raise SystemExit(
                "NEMO_ACT_OFFLOAD=1 with NEMO_RECOMPUTE=full: upstream offloads "
                "module activations that full recompute never stores; use "
                "recompute=none or sel for the offload arm."
            )
        mods = list(offload_modules)
        if recompute == "sel":
            # cannot offload inside a recomputed moe / recomputed modules
            recomp_mods = set(cfg.model.recompute_modules or [])
            if "moe" in recomp_mods:
                mods = [m for m in mods if m not in {"expert_fc1", "moe_act", "fused_group_mlp"}]
            mods = [m for m in mods if m not in recomp_mods]
        cfg.model.offload_modules = mods
        cfg.model.activation_offload_fraction = _env("NEMO_ACT_OFFLOAD_FRACTION", 1.0, float)
    else:
        cfg.model.offload_modules = None

    # LoRA — LF parity: r64/a16/drop0 on every linear incl. one adapter per
    # expert (share_expert_adapters=False), bf16 master. Plain LoRA for VLM
    # checkpoints too (the upstream VL PEFT recipe does the same), but scoped
    # to the language tower: adapters inside the frozen, never-run vision
    # tower would be trainable-yet-gradless and trip DDP's per-param
    # grad-ready accounting. (GDN linear-attention in_proj/out_proj are not
    # in upstream's default target set — leaving them unadapted is strictly
    # generous to NeMo.)
    if is_vlm:
        lora_targets = [
            "*language_model*.linear_qkv",
            "*language_model*.linear_proj",
            "*language_model*.linear_fc1",
            "*language_model*.linear_fc2",
        ]
    else:
        lora_targets = ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"]
    cfg.peft = LoRA(
        target_modules=lora_targets,
        dim=lora_rank,
        alpha=lora_alpha,
        dropout=lora_dropout,
        dropout_position="pre",
        lora_A_init_method="xavier",
        lora_B_init_method="zero",
        lora_dtype=None,
        share_expert_adapters=False,
    )

    # Training length/batch — house protocol: warmup + measured, ga in gbs.
    cfg.train.train_iters = warmup + measured
    cfg.train.micro_batch_size = mbs
    cfg.train.global_batch_size = gbs
    cfg.train.manual_gc = False

    # No validation, no checkpoint saving: this is a throughput/capacity probe.
    cfg.validation.eval_interval = 10_000_000
    cfg.validation.eval_iters = 0
    cfg.checkpoint.save = None
    cfg.checkpoint.load = None
    cfg.checkpoint.save_interval = None
    cfg.checkpoint.pretrained_checkpoint = model_path if load_weights else None

    # Synthetic data pinned at seq_len (the LF driver likewise trains on
    # generated full-length rows). Text models: mock GPT dataset (blend=None
    # => mock mode). VLM checkpoints: text-only mock conversations padded to
    # the full seq — no images, vision tower idle.
    if is_vlm:
        from megatron.bridge.data.vlm_datasets.mock_provider import (
            MockVLMConversationProvider,
        )

        cfg.dataset = MockVLMConversationProvider(
            seq_length=seq_len,
            hf_processor_path=model_path,
            num_images=0,
            random_seed=seed,
            pad_to_max_length=True,
            pad_to_multiple_of=128,
            enable_in_batch_packing=False,
            dataloader_type="single",
        )
    else:
        cfg.dataset = GPTDatasetConfig(
            random_seed=seed,
            reset_attention_mask=False,
            reset_position_ids=False,
            eod_mask_loss=False,
            seq_length=seq_len,
            num_dataset_builder_threads=1,
            blend=None,
            blend_per_split=None,
            split="9999,8,2",
            data_sharding=True,
            dataloader_type="single",
            skip_getting_attention_mask_from_dataset=True,
            # mcore default True materializes a [S,S] numpy causal mask PER
            # SAMPLE in the dataloader worker — 147 GB each at S=384k (host
            # OOM before step 1). TE causal attention needs no mask tensor.
            create_attention_mask=False,
        )

    cfg.logger.log_interval = 1
    cfg.logger.tensorboard_dir = os.path.join(out_dir, "tb_logs")
    cfg.logger.log_timers_to_tensorboard = False

    cfg.scheduler.lr_warmup_iters = 0
    cfg.scheduler.lr_decay_iters = warmup + measured
    cfg.optimizer.lr = _env("NEMO_LR", 1e-4, float)
    cfg.optimizer.min_lr = 0.0

    cfg.mixed_precision = bf16_mixed()

    cfg.rng.seed = seed

    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True
    cfg.ddp.average_in_collective = True
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.check_for_nan_in_grad = False

    rank = int(os.environ.get("RANK", "0"))
    if rank == 0:
        manifest = {
            "model_path": model_path,
            "seq_len": seq_len,
            "micro_batch": mbs,
            "grad_accum": ga,
            "global_batch": gbs,
            "world_size": world,
            "tp": tp,
            "ep": ep,
            "etp": etp,
            "recompute": recompute,
            "act_offload": act_offload,
            "offload_modules": cfg.model.offload_modules,
            "lora": {"rank": lora_rank, "alpha": lora_alpha, "dropout": lora_dropout},
            "warmup": warmup,
            "measured": measured,
            "load_weights": load_weights,
            "dispatcher": dispatcher,
        }
        print("NEMO_RUN_CONFIG " + json.dumps(manifest), flush=True)
        print(
            f"NEMO_DBG_DATASET type={type(cfg.dataset).__name__} "
            f"pad_to_max_length={getattr(cfg.dataset, 'pad_to_max_length', 'n/a')} "
            f"seq_length={getattr(cfg.dataset, 'seq_length', 'n/a')}",
            flush=True,
        )

    finetune(config=cfg, forward_step_func=step_fn)

    import torch.distributed as dist

    peak_alloc = torch.cuda.max_memory_allocated()
    peak_resv = torch.cuda.max_memory_reserved()
    print(
        f"NEMO_PEAK_MEM rank={rank} max_allocated_bytes={peak_alloc} "
        f"max_reserved_bytes={peak_resv}",
        flush=True,
    )
    if dist.is_initialized():
        dist.barrier()
    if rank == 0:
        print("NEMO_RUN_DONE", flush=True)


if __name__ == "__main__":
    main()
