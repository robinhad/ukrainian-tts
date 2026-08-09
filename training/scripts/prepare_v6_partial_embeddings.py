#!/usr/bin/env python3
"""Reuse v5 raw embeddings and merge new v6 clean embeddings."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from kaldiio import ReadHelper, WriteHelper


def read_vectors(root: Path) -> dict[str, np.ndarray]:
    vectors: dict[str, np.ndarray] = {}
    for scp in sorted(root.glob("*/xvector.scp")):
        with ReadHelper(f"scp:{scp}") as reader:
            for key, value in reader:
                if key in vectors:
                    raise RuntimeError(f"Duplicate vector ID: {key}")
                vectors[str(key)] = np.asarray(value, dtype=np.float32).squeeze()
    return vectors


def write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_assignment(target: pd.DataFrame, previous: pd.DataFrame) -> None:
    if len(target) != 74_156 or target["utterance_id"].duplicated().any():
        raise SystemExit("The v6 hybrid manifest must contain 74,156 unique IDs.")
    prior = previous.set_index("utterance_id")
    if set(target["utterance_id"].astype(str)) != set(prior.index.astype(str)):
        raise SystemExit("The v5 and v6 hybrid IDs differ.")
    counts = target["embedding_audio_variant"].value_counts().to_dict()
    if counts != {"raw": 37_078, "clean": 37_078}:
        raise SystemExit(f"The v6 assignment is not exact 50/50: {counts}")
    for row in target.itertuples(index=False):
        identifier = str(row.utterance_id)
        if str(prior.loc[identifier, "embedding_audio_variant"]) != str(
            row.embedding_audio_variant
        ):
            raise SystemExit(f"The embedding assignment changed for {identifier}.")
        if str(row.embedding_audio_variant) == "raw" and Path(
            str(prior.loc[identifier, "embedding_audio_path"])
        ).resolve() != Path(str(row.embedding_audio_path)).resolve():
            raise SystemExit(f"The raw embedding input changed for {identifier}.")


def prepare_clean(args: argparse.Namespace) -> int:
    target = pd.read_parquet(args.hybrid_manifest).sort_values("utterance_id")
    previous = pd.read_parquet(args.v5_hybrid_manifest)
    validate_assignment(target, previous)
    clean = target[target["embedding_audio_variant"] == "clean"]
    archives: dict[str, dict[str, int]] = {}
    for split, part in clean.groupby("split", sort=True):
        directory = args.clean_kaldi_root / str(split)
        speakers: dict[str, list[str]] = defaultdict(list)
        wav_lines = []
        utt2spk = []
        for row in part.itertuples(index=False):
            identifier = str(row.utterance_id)
            speaker = f"{identifier}--{row.speaker_id}"
            wav_lines.append(f"{identifier} {row.embedding_audio_path}")
            utt2spk.append(f"{identifier} {speaker}")
            speakers[speaker].append(identifier)
        write_lines(directory / "wav.scp", sorted(wav_lines))
        write_lines(directory / "utt2spk", sorted(utt2spk))
        write_lines(
            directory / "spk2utt",
            [f"{key} {' '.join(sorted(value))}" for key, value in sorted(speakers.items())],
        )
        archives[str(split)] = {"clean_records": int(len(part))}
    report = {
        "status": "PASS",
        "records": int(len(target)),
        "counts": {"clean": 37_078, "raw": 37_078},
        "assignment_matches_v5": True,
        "raw_input_paths_match_v5": True,
        "clean_sets": archives,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def merge(args: argparse.Namespace) -> int:
    target = pd.read_parquet(args.hybrid_manifest).sort_values("utterance_id")
    previous = pd.read_parquet(args.v5_hybrid_manifest)
    validate_assignment(target, previous)
    old = read_vectors(args.v5_xvector_root)
    clean = read_vectors(args.clean_xvector_root)
    expected_raw = set(
        target.loc[target["embedding_audio_variant"] == "raw", "utterance_id"].astype(str)
    )
    expected_clean = set(
        target.loc[target["embedding_audio_variant"] == "clean", "utterance_id"].astype(str)
    )
    if not expected_raw <= set(old) or set(clean) != expected_clean:
        raise SystemExit("The raw or clean vector archive has incorrect ID coverage.")
    digest = hashlib.sha256()
    archive_report = {}
    for split, part in target.groupby("split", sort=True):
        output = args.output_root / str(split)
        output.mkdir(parents=True, exist_ok=True)
        ark, scp = output / "xvector.ark", output / "xvector.scp"
        with WriteHelper(f"ark,scp:{ark},{scp}") as writer:
            for row in part.itertuples(index=False):
                identifier = str(row.utterance_id)
                vector = old[identifier] if row.embedding_audio_variant == "raw" else clean[identifier]
                if vector.shape != (192,) or not np.isfinite(vector).all() or not np.any(vector):
                    raise SystemExit(f"The vector is invalid for {identifier}.")
                digest.update(identifier.encode() + b"\0" + vector.tobytes())
                writer(identifier, vector)
        archive_report[str(split)] = {"records": int(len(part)), "scp": str(scp.resolve())}
    report = {
        "status": "PASS",
        "records": int(len(target)),
        "counts": {"clean": 37_078, "raw": 37_078},
        "reused_raw_vectors": 37_078,
        "recomputed_clean_vectors": 37_078,
        "assignment_matches_v5": True,
        "raw_input_paths_match_v5": True,
        "dimensions": [192],
        "combined_vector_sha256": digest.hexdigest(),
        "archives": archive_report,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare-clean", "merge"):
        item = sub.add_parser(name)
        item.add_argument("--hybrid-manifest", type=Path, required=True)
        item.add_argument("--v5-hybrid-manifest", type=Path, required=True)
        item.add_argument("--report", type=Path, required=True)
        if name == "prepare-clean":
            item.add_argument("--clean-kaldi-root", type=Path, required=True)
        else:
            item.add_argument("--v5-xvector-root", type=Path, required=True)
            item.add_argument("--clean-xvector-root", type=Path, required=True)
            item.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    return prepare_clean(args) if args.command == "prepare-clean" else merge(args)


if __name__ == "__main__":
    raise SystemExit(main())
