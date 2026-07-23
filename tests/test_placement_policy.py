"""Unit + acceptance tests for asym_gemm.training.placement_policy
(fix_cpu_compute.md item 1; spec: agent/impls/placement.md).

The DRY-RUN ACCEPTANCE HARNESS (test_p10_*) asserts the policy reproduces the
P10 production placement sets EXACTLY from its rules at the three canonical
workloads (30B@32k, 30B@128k, 32B@32k).

Run: .venv/bin/python tests/test_placement_policy.py
"""

import json
import os
import sys
import tempfile

from asym_gemm.training import placement_policy as pol


class _Env:
    """Scoped env patch + policy reset."""

    def __init__(self, **kv):
        self.kv = kv
        self.saved = {}

    def __enter__(self):
        for k, v in self.kv.items():
            self.saved[k] = os.environ.get(k)
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = str(v)
        pol.reset_for_tests()
        return self

    def __exit__(self, *exc):
        for k, v in self.saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        pol.reset_for_tests()


_POLICY_OFF = {"ASYM_PLACEMENT_POLICY": None, "ASYM_GEMM_LF_CONFIG_ASYM_PLACEMENT_POLICY": None}
_POLICY_ON = {"ASYM_PLACEMENT_POLICY": "1", "ASYM_GEMM_LF_CONFIG_ASYM_PLACEMENT_POLICY": None}

# Canonical workload numbers (cpu_compute.md log / placement.md basis):
#   30B@32kxb8 : routed rows 2.1M, act 3.2 GB, per-projection rows 256k
#   30B@128kxb8: routed rows 8.2M, act 12.6 GB, per-projection rows 1.02M
#   32B@32kxb8 : dense; act tensor 13.1 GB class
_W30_32K = dict(rows=2_097_152, nbytes=2 * 768 * 2_097_152, proj_rows=256_000)
_W30_128K = dict(rows=8_192_000, nbytes=2 * 768 * 8_192_000, proj_rows=1_024_000)
_W32_32K = dict(rows=262_144, nbytes=13_107_200_000, proj_rows=256_000)


def _decision_set(rows, nbytes, proj_rows):
    return {
        "P1.moe_cpu_act": pol.moe_cpu_act(rows, nbytes),
        "P2.moe_wgrad_deposit": pol.moe_wgrad_deposit(),
        "P3.attn_wgrad_deposit": pol.attn_wgrad_deposit(proj_rows),
        "P4.boundary_pinned": pol.boundary_pinned(),
        "P5.stage_dedup": pol.stage_dedup(),
        "P6.fused_widen": pol.fused_widen(),
        "P9.moe_cpu_silu_bwd": pol.moe_cpu_silu_bwd(),
        "P9.moe_lora_b_deposit": pol.moe_lora_b_deposit(),
    }


def test_default_off():
    with _Env(**_POLICY_OFF):
        assert pol.enabled() is False


def test_enabled_via_env_and_lf_prefix():
    with _Env(**_POLICY_ON):
        assert pol.enabled() is True
    with _Env(ASYM_PLACEMENT_POLICY=None, ASYM_GEMM_LF_CONFIG_ASYM_PLACEMENT_POLICY="1"):
        assert pol.enabled() is True


def test_thresholds_defaults_and_overrides():
    with _Env(**_POLICY_ON):
        t = pol.thresholds()
        assert t["act_max_rows"] == 4_194_304
        assert t["act_max_bytes"] == 6_400_000_000
        assert t["attn_wgrad_max_rows"] == 262_144
    with _Env(ASYM_POLICY_ACT_MAX_ROWS="100", ASYM_POLICY_ATTN_WGRAD_MAX_ROWS="7", **_POLICY_ON):
        t = pol.thresholds()
        assert t["act_max_rows"] == 100 and t["attn_wgrad_max_rows"] == 7
    # legacy env names still honoured
    with _Env(ASYMM_QWEN3_MOE_FG_CPU_ACT_MAX_ROWS="123", ASYMM_CPU_ACT_MAX_BYTES="456", **_POLICY_ON):
        t = pol.thresholds()
        assert t["act_max_rows"] == 123 and t["act_max_bytes"] == 456


def test_p1_rows_and_bytes_boundaries():
    with _Env(**_POLICY_ON):
        pol.register_model_class("moe")
        t = pol.thresholds()
        assert pol.moe_cpu_act(t["act_max_rows"], 1) is True
        assert pol.moe_cpu_act(t["act_max_rows"] + 1, 1) is False
        assert pol.moe_cpu_act(1, t["act_max_bytes"]) is True
        assert pol.moe_cpu_act(1, t["act_max_bytes"] + 1) is False


def test_p3_rows_boundary():
    with _Env(**_POLICY_ON):
        pol.register_model_class("moe")
        t = pol.thresholds()
        assert pol.attn_wgrad_deposit(t["attn_wgrad_max_rows"]) is True
        assert pol.attn_wgrad_deposit(t["attn_wgrad_max_rows"] + 1) is False


def test_p8_dense_kills_cpu_compute():
    with _Env(**_POLICY_ON):
        pol.register_model_class("dense")
        assert pol.moe_cpu_act(1, 1) is False
        assert pol.moe_cpu_act_feature() is False
        assert pol.moe_wgrad_deposit() is False
        assert pol.attn_wgrad_feature() is False
        assert pol.attn_wgrad_deposit(1) is False
        assert pol.dense_cpu_compute() is False
        # never-hurts class stays ON on dense
        assert pol.boundary_pinned() is True
        assert pol.fused_widen() is True


def test_dense_is_sticky_over_moe():
    with _Env(**_POLICY_ON):
        pol.register_model_class("moe")
        pol.register_model_class("dense")
        assert pol.model_class() == "dense"
        assert pol.moe_wgrad_deposit() is False


def test_unknown_model_class_is_conservative_for_deposits():
    with _Env(**_POLICY_ON):
        assert pol.model_class() == "unknown"
        assert pol.attn_wgrad_feature() is False
        assert pol.attn_wgrad_deposit(1) is False
        assert pol.moe_wgrad_deposit() is False


# ---------------- P10 dry-run acceptance harness ----------------


def test_p10_moe_32k_production_set():
    with _Env(**_POLICY_ON):
        pol.register_model_class("moe")
        d = _decision_set(**_W30_32K)
        assert d == {
            "P1.moe_cpu_act": True,          # CPU act (async, chunked) engages
            "P2.moe_wgrad_deposit": True,    # MoE dA deposit
            "P3.attn_wgrad_deposit": True,   # attention dA deposit (256k rows fit)
            "P4.boundary_pinned": True,
            "P5.stage_dedup": True,
            "P6.fused_widen": True,
            "P9.moe_cpu_silu_bwd": False,    # rejected placements stay off
            "P9.moe_lora_b_deposit": False,
        }, f"30B@32k set mismatch: {d}"
        assert pol.moe_cpu_act_async() and pol.moe_cpu_act_chunked()
        assert pol.cpu_worker() and pol.cpu_worker_bg() and pol.fused_silu_kernels_enabled()


def test_p10_moe_128k_production_set():
    with _Env(**_POLICY_ON):
        pol.register_model_class("moe")
        d = _decision_set(**_W30_128K)
        assert d == {
            "P1.moe_cpu_act": False,         # rows 8.2M > 4.2M -> auto-off
            "P2.moe_wgrad_deposit": True,    # the only deposit that survives 128k
            "P3.attn_wgrad_deposit": False,  # proj rows 1.02M > 256k -> auto-off
            "P4.boundary_pinned": True,
            "P5.stage_dedup": True,
            "P6.fused_widen": True,
            "P9.moe_cpu_silu_bwd": False,
            "P9.moe_lora_b_deposit": False,
        }, f"30B@128k set mismatch: {d}"


def test_p10_dense_32k_production_set():
    with _Env(**_POLICY_ON):
        pol.register_model_class("dense")
        d = _decision_set(**_W32_32K)
        assert d == {
            "P1.moe_cpu_act": False,         # P8 regime: all CPU compute off
            "P2.moe_wgrad_deposit": False,
            "P3.attn_wgrad_deposit": False,
            "P4.boundary_pinned": True,      # never-hurts only
            "P5.stage_dedup": True,          # (unreachable on dense; MoE-path only)
            "P6.fused_widen": True,
            "P9.moe_cpu_silu_bwd": False,
            "P9.moe_lora_b_deposit": False,
        }, f"32B@32k set mismatch: {d}"


# ---------------- gate-helper integration (env ignored under policy) ----------------


def test_gate_helpers_follow_policy_not_env():
    from asym_gemm.training import qwen3_moe_finegrained as fg
    from asym_gemm.training import dense_mlp_finegrained as dfg
    from asym_gemm.training import attention_activation_offload as aao
    from asym_gemm.training import cpu_worker

    # policy OFF -> env flags rule
    with _Env(ASYMM_QWEN3_MOE_FG_CPU_ACT="1", ASYMM_LORA_A_GRAD_CPU="1",
              ASYMM_ATTN_LORA_A_GRAD_CPU="1", ASYMM_QWEN3_MOE_FG_CPU_SILU_BWD="1",
              ASYMM_CPU_WORKER=None, **_POLICY_OFF):
        assert fg._fg_cpu_act_enabled() is True
        assert fg._lora_a_grad_cpu_deposit_enabled() is True
        assert aao._attn_lora_a_grad_cpu_deposit_enabled() is True
        assert fg._fg_cpu_silu_bwd_enabled() is True
        assert cpu_worker.enabled() is False
        assert dfg._dense_lora_a_grad_cpu_deposit_enabled() is True
        # legacy per-call guard still enforced with policy OFF
        assert fg._cpu_act_fits(4_194_304, 1) is True
        assert fg._cpu_act_fits(4_194_305, 1) is False

    # policy ON + MoE -> production decisions regardless of env flags
    with _Env(ASYMM_QWEN3_MOE_FG_CPU_ACT=None, ASYMM_LORA_A_GRAD_CPU=None,
              ASYMM_ATTN_LORA_A_GRAD_CPU=None, ASYMM_QWEN3_MOE_FG_CPU_SILU_BWD="1",
              ASYMM_LORA_B_GRAD_CPU="1", ASYMM_QWEN3_MOE_FG_STAGE_DEDUP=None,
              ASYMM_QWEN3_MOE_FG_CPU_ACT_CHUNKED=None, ASYMM_CPU_WORKER=None,
              ASYM_CPU_WORKER_BG=None, **_POLICY_ON):
        pol.register_model_class("moe")
        assert fg._fg_cpu_act_enabled() is True          # P1 (flags unset)
        assert fg._fg_cpu_act_async_enabled() is True
        assert fg._fg_cpu_act_chunked_enabled() is True  # K-4 rides P1
        assert fg._fg_stage_dedup_enabled() is True      # P5
        assert fg._lora_a_grad_cpu_deposit_enabled() is True   # P2
        assert aao._attn_lora_a_grad_cpu_deposit_enabled() is True  # P3 feature
        assert fg._fg_cpu_silu_bwd_enabled() is False    # P9 beats env=1
        assert fg._fg_lora_b_grad_cpu_deposit_enabled() is False  # P9 beats env=1
        assert cpu_worker.enabled() is True              # implied infra
        assert cpu_worker.bg_enabled() is True
        assert fg._cpu_act_fits(2_097_152, 3_221_225_472) is True
        assert fg._cpu_act_fits(8_192_000, 12_582_912_000) is False

    # policy ON + dense -> P8 kill-switch beats env flags
    with _Env(ASYMM_DENSE_MLP_FINEGRAINED_CPU_ACT="1", ASYMM_LORA_A_GRAD_CPU="1",
              ASYMM_ATTN_LORA_A_GRAD_CPU="1", ASYMM_DENSE_MLP_FG_CPU_ACT_ASYNC="1",
              **_POLICY_ON):
        pol.register_model_class("dense")
        assert dfg._finegrained_cpu_activation_enabled() is False
        assert dfg._dense_cpu_act_async_enabled() is False
        assert dfg._dense_lora_a_grad_cpu_deposit_enabled() is False
        assert aao._attn_lora_a_grad_cpu_deposit_enabled() is False
        assert fg._lora_a_grad_cpu_deposit_enabled() is False


def test_trace_and_sidecar_export():
    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, "source_profile.json")
        with _Env(ASYM_GEMM_LF_PROFILE_SOURCE_JSON=src, **_POLICY_ON):
            pol.register_model_class("moe")
            pol.moe_cpu_act(**{k: _W30_32K[k] for k in ("rows", "nbytes")})
            pol.moe_cpu_act(**{k: _W30_128K[k] for k in ("rows", "nbytes")})
            pol.attn_wgrad_deposit(_W30_32K["proj_rows"])
            pol.moe_wgrad_deposit()
            pol.boundary_pinned()
            s = pol.stats()
            assert s["enabled"] is True and s["model_class"] == "moe"
            rules = s["rules"]
            assert rules["P1.moe_cpu_act"]["decisions"]["True"]["count"] == 1
            assert rules["P1.moe_cpu_act"]["decisions"]["False"]["count"] == 1
            assert rules["P1.moe_cpu_act"]["decisions"]["False"]["example_inputs"]["rows"] == _W30_128K["rows"]
            assert rules["P3.attn_wgrad_deposit"]["decisions"]["True"]["count"] == 1
            path = os.path.join(td, "placement_policy.json")
            assert os.path.exists(path), "sidecar placement_policy.json not written"
            with open(path) as f:
                payload = json.load(f)
            assert payload["enabled"] is True
            assert "P2.moe_wgrad_deposit" in payload["rules"]
            assert "P4.boundary_pinned" in payload["rules"]


def main():
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            import traceback

            print(f"FAIL {name}: {exc}")
            traceback.print_exc()
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
