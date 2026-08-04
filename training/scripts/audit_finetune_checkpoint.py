#!/usr/bin/env python3
"""Check the step state and finite model values in a GAN-TTS checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
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
    status = (
        "PASS"
        if reported_steps == args.expected_steps
        and optimizer_steps == {args.expected_steps}
        and not nonfinite
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
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
