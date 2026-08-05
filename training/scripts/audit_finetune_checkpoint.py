#!/usr/bin/env python3
"""Check the step state and finite model values in a GAN-TTS checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--model-artifact",
        type=Path,
        help="Optional weight-only milestone that must equal the checkpoint model.",
    )
    parser.add_argument("--expected-steps", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    reporter = payload.get("reporter", {})
    epoch = int(reporter.get("epoch", 0))
    stats = reporter.get("stats", {}).get(epoch, {})
    reported_steps = int(stats.get("train", {}).get("total_count", 0))
    optimizer_steps = {
        int(state["step"])
        for optimizer in payload.get("optimizers", [])
        for state in optimizer.get("state", {}).values()
        if isinstance(state, dict) and "step" in state
    }
    model = payload.get("model", {})
    nonfinite = [
        name
        for name, value in model.items()
        if torch.is_tensor(value)
        and (value.is_floating_point() or value.is_complex())
        and not torch.isfinite(value).all()
    ]
    model_artifact_report = None
    model_artifact_matches = True
    if args.model_artifact is not None:
        artifact_payload = torch.load(
            args.model_artifact, map_location="cpu", weights_only=False
        )
        artifact_model = artifact_payload.get("model", artifact_payload)
        missing_keys = sorted(set(model) - set(artifact_model))
        unexpected_keys = sorted(set(artifact_model) - set(model))
        different_keys = sorted(
            name
            for name in set(model) & set(artifact_model)
            if not torch.equal(model[name], artifact_model[name])
        )
        artifact_nonfinite = [
            name
            for name, value in artifact_model.items()
            if torch.is_tensor(value)
            and (value.is_floating_point() or value.is_complex())
            and not torch.isfinite(value).all()
        ]
        model_artifact_matches = not (
            missing_keys or unexpected_keys or different_keys or artifact_nonfinite
        )
        model_artifact_report = {
            "path": str(args.model_artifact.resolve()),
            "sha256": hashlib.sha256(args.model_artifact.read_bytes()).hexdigest(),
            "model_tensor_count": sum(
                torch.is_tensor(value) for value in artifact_model.values()
            ),
            "missing_keys": missing_keys,
            "unexpected_keys": unexpected_keys,
            "different_keys": different_keys,
            "nonfinite_model_tensors": artifact_nonfinite,
            "matches_checkpoint_model": model_artifact_matches,
        }

    status = (
        "PASS"
        if reported_steps == args.expected_steps
        and optimizer_steps == {args.expected_steps}
        and not nonfinite
        and model_artifact_matches
        else "FAIL"
    )
    report = {
        "status": status,
        "checkpoint": str(args.checkpoint.resolve()),
        "epoch": epoch,
        "reported_steps": reported_steps,
        "expected_steps": args.expected_steps,
        "optimizer_steps": sorted(optimizer_steps),
        "model_tensor_count": sum(torch.is_tensor(value) for value in model.values()),
        "nonfinite_model_tensors": nonfinite,
        "model_artifact": model_artifact_report,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
