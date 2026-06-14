from __future__ import annotations

import pickle
from pathlib import Path

from scripts.testing.analyze_cuda_memory_snapshot import analyze_snapshot, main


def _frame(filename: str, name: str = "forward", line: int = 1) -> dict:
    return {"filename": filename, "name": name, "line": line}


def test_analyzer_replays_device_traces_and_attributes_peak(tmp_path: Path) -> None:
    snapshot = {
        "device_traces": [
            [
                {
                    "action": "Action.ALLOC",
                    "addr": 0x1000,
                    "size": 100,
                    "frames": [_frame("/repo/asym_gemm/training/qwen3_moe.py", "_ActivationOffloadQwen3ExpertFunction.forward", 10)],
                },
                {
                    "action": "alloc",
                    "addr": 0x2000,
                    "size": 200,
                    "frames": [_frame("/repo/modeling_qwen3.py", "self_attn_forward", 20)],
                },
                {"action": "free_requested", "addr": 0x1000},
                {"action": "free_completed", "addr": 0x1000},
                {
                    "action": "alloc",
                    "addr": 0x3000,
                    "size": 300,
                    "frames": [_frame("/repo/loss.py", "cross_entropy_loss", 30)],
                },
            ]
        ]
    }

    report = analyze_snapshot(snapshot, device=0, top=10)

    assert report["peak_live_bytes"] == 500
    assert report["final_live_bytes"] == 500
    assert report["unknown_free_events"] == 0
    assert {row["component"]: row["bytes"] for row in report["bucket_rows"]} == {
        "loss": 300,
        "attention": 200,
    }
    assert report["top_blocks"][0]["bytes"] == 300
    assert report["top_blocks"][0]["component"] == "loss"


def test_analyzer_cli_writes_json_and_markdown(tmp_path: Path) -> None:
    snapshot_path = tmp_path / "memory_snapshot.pickle"
    output_json = tmp_path / "out.json"
    output_md = tmp_path / "out.md"
    snapshot = {
        "device_traces": [
            [
                {
                    "action": "alloc",
                    "addr": 0x10,
                    "size": 123,
                    "frames": [_frame("/repo/asym_gemm/training/moe.py", "scatter_contiguous", 7)],
                }
            ]
        ]
    }
    with snapshot_path.open("wb") as handle:
        pickle.dump(snapshot, handle)

    rc = main(
        [
            "--snapshot",
            str(snapshot_path),
            "--device",
            "0",
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ]
    )

    assert rc == 0
    assert '"peak_live_bytes": 123' in output_json.read_text(encoding="utf-8")
    assert "routed_experts" in output_md.read_text(encoding="utf-8")
