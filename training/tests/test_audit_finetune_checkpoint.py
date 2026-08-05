import json
import subprocess
import sys
from pathlib import Path

import torch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_finetune_checkpoint.py"


def test_audit_finetune_checkpoint_passes_new_step_state(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pth"
    output = tmp_path / "report.json"
    torch.save(
        {
            "model": {"generator.weight": torch.ones(2)},
            "optimizers": [{"state": {0: {"step": torch.tensor(100)}}}],
            "reporter": {
                "epoch": 1,
                "stats": {1: {"train": {"total_count": 100}}},
            },
        },
        checkpoint,
    )
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--checkpoint",
            str(checkpoint),
            "--expected-steps",
            "100",
            "--output",
            str(output),
        ],
        check=True,
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "PASS"
    assert report["optimizer_steps"] == [100]


def test_audit_finetune_checkpoint_checks_weight_only_artifact(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint.pth"
    artifact = tmp_path / "100step.pth"
    output = tmp_path / "report.json"
    model = {"generator.weight": torch.ones(2)}
    torch.save(
        {
            "model": model,
            "optimizers": [{"state": {0: {"step": torch.tensor(100)}}}],
            "reporter": {
                "epoch": 1,
                "stats": {1: {"train": {"total_count": 100}}},
            },
        },
        checkpoint,
    )
    torch.save(model, artifact)
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--checkpoint",
            str(checkpoint),
            "--model-artifact",
            str(artifact),
            "--expected-steps",
            "100",
            "--output",
            str(output),
        ],
        check=True,
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "PASS"
    assert report["model_artifact"]["matches_checkpoint_model"] is True
