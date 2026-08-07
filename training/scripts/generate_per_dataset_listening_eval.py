#!/usr/bin/env python3
"""Make a balanced reference and synthesis listening set for each dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from training.frontend.phonemize import UkrainianPhonemizer


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evenly_spaced_rows(frame: pd.DataFrame, count: int) -> pd.DataFrame:
    """Select deterministic rows across the duration range."""
    if count < 1 or count > len(frame):
        raise ValueError(f"count must be in the range 1..{len(frame)}")
    ordered = frame.sort_values(["duration", "utterance_id"], kind="stable")
    if count == 1:
        return ordered.iloc[[len(ordered) // 2]]
    indices = [round(i * (len(ordered) - 1) / (count - 1)) for i in range(count)]
    return ordered.iloc[indices]


def select_rows(frame: pd.DataFrame, count_per_dataset: int) -> pd.DataFrame:
    """Select an equal pre-trim and post-trim embedding mix per dataset."""
    required = {
        "utterance_id",
        "source_id",
        "duration",
        "embedding_audio_variant",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"manifest columns are missing: {missing}")
    if count_per_dataset < 2 or count_per_dataset % 2:
        raise ValueError("count_per_dataset must be an even number of at least 2")

    selected = []
    count_per_variant = count_per_dataset // 2
    for source_id, source_frame in frame.groupby("source_id", sort=True):
        source_rows = []
        for variant in ("raw", "clean"):
            variant_frame = source_frame.loc[
                source_frame["embedding_audio_variant"] == variant
            ]
            if len(variant_frame) < count_per_variant:
                raise ValueError(
                    f"{source_id} has {len(variant_frame)} {variant} rows; "
                    f"it needs {count_per_variant}"
                )
            source_rows.append(evenly_spaced_rows(variant_frame, count_per_variant))
        source_selected = pd.concat(source_rows, ignore_index=True).sort_values(
            ["duration", "embedding_audio_variant", "utterance_id"], kind="stable"
        )
        source_selected = source_selected.copy()
        source_selected["listening_index"] = range(1, count_per_dataset + 1)
        selected.append(source_selected)
    return pd.concat(selected, ignore_index=True)


def make_link(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    destination.symlink_to(
        os.path.relpath(source.resolve(), destination.parent.resolve())
    )


def validate_audio(path: Path) -> dict:
    waveform, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    info = sf.info(path)
    errors = []
    if sample_rate != 24000:
        errors.append(f"sample_rate={sample_rate}")
    if info.channels != 1 or waveform.shape[1] != 1:
        errors.append(f"channels={info.channels}")
    if waveform.size == 0 or info.frames == 0:
        errors.append("empty waveform")
    if not np.isfinite(waveform).all():
        errors.append("non-finite waveform")
    peak = float(np.max(np.abs(waveform))) if waveform.size else 0.0
    if peak == 0.0:
        errors.append("all-zero waveform")
    return {
        "sample_rate": int(sample_rate),
        "channels": int(info.channels),
        "duration": float(info.frames / sample_rate) if sample_rate else 0.0,
        "peak_absolute": peak,
        "possible_clipping": peak >= 0.999,
        "errors": errors,
    }


def load_embedding_loaders(xvector_root: Path, splits: set[str]) -> dict:
    from kaldiio import load_scp

    loaders = {}
    for split in sorted(splits):
        scp = xvector_root / split / "xvector.scp"
        if not scp.is_file():
            raise FileNotFoundError(scp)
        loaders[split] = load_scp(str(scp))
    return loaders


def write_tsv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--xvector-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--count-per-dataset", type=int, default=10)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()

    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError(f"output directory is not empty: {args.output}")
    frame = pd.read_parquet(args.manifest)
    selected = select_rows(frame, args.count_per_dataset)
    loaders = load_embedding_loaders(
        args.xvector_root, set(map(str, selected["split"]))
    )

    from espnet2.bin.tts_inference import Text2Speech

    frontend = UkrainianPhonemizer()
    model = Text2Speech(
        train_config=args.config,
        model_file=args.checkpoint,
        device=args.device,
    )
    if int(model.fs) != 24000:
        raise RuntimeError(f"model sample rate is {model.fs}, expected 24000")

    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint_hash = sha256(args.checkpoint)
    config_hash = sha256(args.config)
    records = []
    errors = []
    for row in selected.itertuples(index=False):
        source_id = str(row.source_id)
        utterance_id = str(row.utterance_id)
        index = int(row.listening_index)
        split = str(row.split)
        vector = np.asarray(loaders[split][utterance_id], dtype=np.float32).squeeze()
        if vector.shape != (192,) or not np.isfinite(vector).all() or not np.any(vector):
            raise RuntimeError(f"{utterance_id} has an invalid speaker embedding")

        text_raw = str(row.text_raw)
        sanitized, tokens = frontend.phonemize(text_raw)
        started = time.monotonic()
        result = model(sanitized, spembs=vector)
        generation_seconds = time.monotonic() - started
        waveform = result["wav"].view(-1).detach().cpu().numpy().astype(np.float32)
        if waveform.size == 0 or not np.isfinite(waveform).all() or not np.any(waveform):
            raise RuntimeError(f"{utterance_id} has an invalid synthesized waveform")

        item_name = f"{index:02d}_{utterance_id}"
        source_root = args.output / source_id
        synthesis_path = source_root / "synthesized_100k" / f"{item_name}.wav"
        reference_path = source_root / "reference_trim_only" / f"{item_name}.wav"
        synthesis_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(synthesis_path, waveform, 24000, subtype="PCM_16")
        make_link(Path(row.audio_path), reference_path)

        reference_check = validate_audio(reference_path)
        synthesis_check = validate_audio(synthesis_path)
        item_errors = [
            *(f"reference: {value}" for value in reference_check["errors"]),
            *(f"synthesis: {value}" for value in synthesis_check["errors"]),
        ]
        if item_errors:
            errors.append({"utterance_id": utterance_id, "errors": item_errors})
        records.append(
            {
                "dataset": source_id,
                "listening_index": index,
                "utterance_id": utterance_id,
                "split": split,
                "embedding_audio_variant": str(row.embedding_audio_variant),
                "text_raw": text_raw,
                "text_sanitized": sanitized,
                "espeak_phonemes": tokens,
                "reference": str(reference_path.resolve()),
                "reference_link": str(reference_path),
                "reference_check": reference_check,
                "synthesis": str(synthesis_path.resolve()),
                "synthesis_check": synthesis_check,
                "generation_seconds": generation_seconds,
                "real_time_factor": generation_seconds
                / synthesis_check["duration"],
                "speaker_embedding_dimension": int(vector.size),
                "frontend_config_hash": frontend.config.digest,
                "espeak_version": frontend.config.espeak_version,
                "checkpoint_sha256": checkpoint_hash,
                "config_sha256": config_hash,
            }
        )

    tsv_rows = [
        {
            "dataset": item["dataset"],
            "item": f"{item['listening_index']:02d}",
            "utterance_id": item["utterance_id"],
            "embedding_variant": item["embedding_audio_variant"],
            "duration_seconds": f"{item['synthesis_check']['duration']:.3f}",
            "text": item["text_sanitized"],
            "reference": str(
                Path(item["reference_link"]).relative_to(args.output)
            ),
            "synthesized_100k": str(
                Path(item["synthesis"]).relative_to(args.output.resolve())
            ),
        }
        for item in records
    ]
    manifest_fields = [
        "dataset",
        "item",
        "utterance_id",
        "embedding_variant",
        "duration_seconds",
        "text",
        "reference",
        "synthesized_100k",
    ]
    write_tsv(args.output / "manifest.tsv", tsv_rows, manifest_fields)
    feedback_fields = [
        *manifest_fields,
        "reference_quality_1_to_5",
        "synthesis_quality_1_to_5",
        "metallic_yes_no",
        "rasp_yes_no",
        "robotic_yes_no",
        "pronunciation_issue",
        "comments",
    ]
    write_tsv(args.output / "feedback_template.tsv", tsv_rows, feedback_fields)

    by_dataset = {}
    for source_id, group in selected.groupby("source_id", sort=True):
        by_dataset[str(source_id)] = {
            "items": int(len(group)),
            "raw_embedding_items": int(
                (group["embedding_audio_variant"] == "raw").sum()
            ),
            "clean_embedding_items": int(
                (group["embedding_audio_variant"] == "clean").sum()
            ),
        }
    report = {
        "status": "PASS" if not errors else "FAIL",
        "manifest": str(args.manifest.resolve()),
        "output": str(args.output.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_hash,
        "config": str(args.config.resolve()),
        "config_sha256": config_hash,
        "dataset_count": len(by_dataset),
        "items_per_dataset": args.count_per_dataset,
        "synthesis_count": len(records),
        "reference_count": len(records),
        "datasets": by_dataset,
        "errors": errors,
        "records": records,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    readme = f"""# Per-Dataset Listening Set

This document uses ASD-STE100 Simplified Technical English style. An approved
STE checker did not certify this document.

This set has {len(by_dataset)} datasets. Each dataset has
{args.count_per_dataset} synthesized WAV files and {args.count_per_dataset}
trim-only reference WAV files. Five items use pre-trim speaker embeddings.
Five items use post-trim speaker embeddings.

Open `manifest.tsv` to see the text and file paths. Listen to the reference
first. Then listen to the 100K synthesis. Enter the results in
`feedback_template.tsv`.

Use a value from 1 to 5 for quality. Use `yes` or `no` for metallic sound,
rasp, and robotic sound. Add a short note for a pronunciation problem.

The automatic validation status is `{report['status']}`. Automatic validation
does not measure naturalness.
"""
    (args.output / "README.md").write_text(readme, encoding="utf-8")
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "records"},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
