from argparse import Namespace
from pathlib import Path

from scripts.plotting.plot_activation_recompute_sweep import (
    parse_job_dir_parts as parse_activation_job_dir_parts,
    parse_expert_policy_spec,
    parse_result_dir,
    passes_filters,
)
from scripts.plotting.plot_lf_interconnect_ctc import _parse_job_dir_parts as parse_ctc_job_dir_parts
from scripts.plotting.plot_lf_memory_breakdown import _parse_job_dir_parts as parse_memory_job_dir_parts
from scripts.plotting.plot_lf_utilization import _parse_job_dir_parts as parse_utilization_job_dir_parts


def _filter_args(policy: str) -> Namespace:
    return Namespace(
        precision="",
        workload=[],
        backend=[],
        router_mode=[],
        expact=[],
        attnact=[],
        layeract=[],
        layergc=[],
        profiler=[],
        liger_loss=[],
        recompute=[],
        workload_tuples=set(),
        expert_recompute_policies=[policy],
    )


def test_activation_recompute_plot_policy_parser_accepts_lf_gc_labels() -> None:
    gc_attn = parse_expert_policy_spec("gc-attn-exp")
    assert gc_attn["expert_recompute_policy_spec"] == "gc-attn-exp"
    assert gc_attn["expert_policy_label"] == "gc-attn-exp"
    assert gc_attn["expert_recompute_policy"] == "gc"
    assert gc_attn["expert_recompute_impl"] == "torch_checkpoint"

    gc_layer = parse_expert_policy_spec("gc-layer")
    assert gc_layer["expert_recompute_policy_spec"] == "gc-layer"
    assert gc_layer["expert_policy_label"] == "gc-layer"
    assert gc_layer["expert_recompute_policy"] == "none"
    assert gc_layer["expert_recompute_impl"] == "none"


def test_activation_recompute_plot_policy_filter_matches_gc_attn_folder_label() -> None:
    result_dir = Path(
        "profiling_results/profiling_both/asym_long_sft_smoke__lora__lf__bf16/"
        "qwen3_5-35b-a3b__gpus1__b2_s2048_ga1_w1_s1_r64_a16_drop000/"
        "asym_cpuadamwds__source__norecomp__polgc-attn-exp__routerwhole__"
        "expact0__attnact0__layeract0__layergc0__loraafwdhbm__actrecomp0__"
        "xunpack0__ligerloss0__gradofftrue__weightofftrue/"
        "b2_s2048_ga1"
    )
    meta = parse_result_dir(result_dir)

    assert meta is not None
    assert meta["expert_recompute_policy_spec"] == "gc-attn-exp"
    assert passes_filters(_filter_args("gc-attn-exp"), meta)


def test_activation_recompute_plot_filters_layer_gc_axis() -> None:
    result_dir = Path(
        "profiling_results/profiling_both/asym_long_sft_smoke__lora__lf__bf16/"
        "qwen3_5-35b-a3b__gpus1__b2_s2048_ga1_w1_s1_r64_a16_drop000/"
        "asym_cpuadamwds__source__norecomp__polnone__routerwhole__"
        "expact1__attnact1__layeract0__layergc1__loraafwdhbm__actrecomp0__"
        "xunpack0__ligerloss0__gradofftrue__weightofftrue/"
        "b2_s2048_ga1"
    )
    meta = parse_result_dir(result_dir)

    assert meta is not None
    assert meta["layeract"] == "layeract0"
    assert meta["layergc"] == "layergc1"

    matching_args = _filter_args("none")
    matching_args.layeract = ["layeract0"]
    matching_args.layergc = ["layergc1"]
    assert passes_filters(matching_args, meta)

    mismatched_args = _filter_args("none")
    mismatched_args.layergc = ["layergc0"]
    assert not passes_filters(mismatched_args, meta)


def test_lf_plotters_ignore_unknown_config_axes_in_job_dir_tail() -> None:
    job_dir = (
        "asym_cpuadamwds__source__norecomp__polgc-attn-exp__routernewmode__"
        "expact0__attnact0__layeract0__layergc0__futureaxis42__loraafwdhbm__"
        "actrecomp0__xunpack0__ligerloss0__gradofftrue__weightofftrue"
    )

    activation_meta = parse_activation_job_dir_parts(job_dir)
    memory_meta = parse_memory_job_dir_parts(job_dir)
    ctc_meta = parse_ctc_job_dir_parts(job_dir)
    utilization_meta = parse_utilization_job_dir_parts(job_dir)

    for meta in (activation_meta, memory_meta, ctc_meta, utilization_meta):
        assert meta is not None
        assert meta["policy_part"] == "polgc-attn-exp"
        assert meta.get("router_mode", meta.get("router_part", "")) in {"newmode", "routernewmode"}
        assert meta["expact"] == "expact0"
        assert meta["attnact"] == "attnact0"
        assert meta["layeract"] == "layeract0"
        assert meta["layergc"] == "layergc0"
        assert meta["liger_loss"] == "ligerloss0"
