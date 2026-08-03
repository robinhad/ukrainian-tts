#!/usr/bin/env python3
"""Preserve validation-best checkpoints before epoch retention removes them."""

from __future__ import annotations

import argparse
import errno
import json
import os
import re
import shutil
import tempfile
import time
from pathlib import Path


METRICS = (
    "generator_loss",
    "generator_g_mel_loss",
    "generator_align_loss",
)
EPOCH_PATTERN = re.compile(r"INFO: (\d+)epoch results:")


def parse_validation(log_path: Path) -> dict[int, dict[str, float]]:
    rows: dict[int, dict[str, float]] = {}
    with log_path.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            match = EPOCH_PATTERN.search(line)
            if match is None or "[valid]" not in line:
                continue
            values = line.split("[valid]", 1)[1]
            row: dict[str, float] = {}
            for metric in METRICS:
                value = re.search(
                    rf"(?<![A-Za-z_]){re.escape(metric)}=([0-9.eE+-]+)",
                    values,
                )
                if value is not None:
                    row[metric] = float(value.group(1))
            if len(row) == len(METRICS):
                rows[int(match.group(1))] = row
    return rows


def checkpoint_source(exp_dir: Path, best_dir: Path, epoch: int) -> Path | None:
    for path in (exp_dir / f"{epoch}epoch.pth", best_dir / f"{epoch}epoch.pth"):
        if path.is_file() and path.stat().st_size > 0:
            return path
    return None


def preserve_checkpoint(source: Path, target: Path) -> str:
    if target.exists():
        return "existing"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".part")
    temporary.unlink(missing_ok=True)
    try:
        os.link(source, temporary)
        method = "hardlink"
    except OSError as error:
        if error.errno != errno.EXDEV:
            raise
        shutil.copy2(source, temporary)
        method = "copy"
    if temporary.stat().st_size != source.stat().st_size:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Checkpoint size changed during preservation: {source}")
    os.replace(temporary, target)
    return method


def write_state(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".part", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def preserve_best(
    log_path: Path,
    exp_dir: Path,
    keep: int,
) -> dict[str, object]:
    best_dir = exp_dir / "best_checkpoints"
    rows = parse_validation(log_path)
    selected: dict[str, list[dict[str, object]]] = {}
    preserved: dict[int, str] = {}

    for metric in METRICS:
        available = []
        for epoch, values in rows.items():
            source = checkpoint_source(exp_dir, best_dir, epoch)
            if source is not None:
                available.append((values[metric], epoch, source))
        ranked = sorted(available)[:keep]
        selected[metric] = []
        for value, epoch, source in ranked:
            target = best_dir / f"{epoch}epoch.pth"
            method = preserve_checkpoint(source, target)
            preserved.setdefault(epoch, method)
            selected[metric].append(
                {
                    "epoch": epoch,
                    "value": value,
                    "checkpoint": str(target),
                }
            )

    selected_epochs = {
        int(item["epoch"])
        for values in selected.values()
        for item in values
    }
    removed_epochs = []
    for target in best_dir.glob("*epoch.pth"):
        match = re.fullmatch(r"(\d+)epoch\.pth", target.name)
        if match is None:
            continue
        epoch = int(match.group(1))
        if epoch not in selected_epochs:
            target.unlink()
            removed_epochs.append(epoch)

    payload: dict[str, object] = {
        "log": str(log_path),
        "experiment": str(exp_dir),
        "keep_per_metric": keep,
        "metrics": selected,
        "preserved_epochs": sorted(preserved),
        "removed_epochs": sorted(removed_epochs),
        "preservation_methods": {
            str(epoch): preserved[epoch] for epoch in sorted(preserved)
        },
        "updated_unix": time.time(),
    }
    write_state(best_dir / "validation_best.json", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--exp-dir", type=Path, required=True)
    parser.add_argument("--keep", type=int, default=3)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--target-epoch", type=int, default=500)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    if args.keep < 1:
        parser.error("--keep must be positive")
    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be positive")
    if not args.log.is_file():
        parser.error(f"Log does not exist: {args.log}")
    if not args.exp_dir.is_dir():
        parser.error(f"Experiment directory does not exist: {args.exp_dir}")

    while True:
        payload = preserve_best(args.log, args.exp_dir, args.keep)
        metrics = payload["metrics"]
        summary = " ".join(
            f"{metric}="
            + ",".join(
                f"e{item['epoch']}:{item['value']:.3f}" for item in metrics[metric]
            )
            for metric in METRICS
        )
        print(f"validation_best {summary}", flush=True)
        if args.once:
            return 0
        target = args.exp_dir / "best_checkpoints" / f"{args.target_epoch}epoch.pth"
        source = args.exp_dir / f"{args.target_epoch}epoch.pth"
        if target.is_file() or source.is_file():
            preserve_best(args.log, args.exp_dir, args.keep)
            return 0
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
