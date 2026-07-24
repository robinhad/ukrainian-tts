#!/usr/bin/env python3
"""Select deterministic, diverse ECAPA vectors for a listening evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from kaldiio import load_ark, save_ark
from sklearn.cluster import MiniBatchKMeans


def unit_rows(values: np.ndarray) -> np.ndarray:
    """Return row vectors with unit L2 norm."""
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if not np.isfinite(values).all() or np.any(norms <= 0):
        raise ValueError("speaker vectors must be finite and nonzero")
    return values / norms


def representative_cluster_medoids(
    candidates: np.ndarray, count: int
) -> list[int]:
    """Select one representative vector from each deterministic cluster."""
    candidate_units = unit_rows(candidates)
    if count < 1 or candidate_units.shape[0] < count:
        raise ValueError("not enough candidate vectors")
    model = MiniBatchKMeans(
        n_clusters=count,
        random_state=0,
        batch_size=2048,
        n_init=10,
        max_iter=200,
    )
    labels = model.fit_predict(candidate_units)
    centers = unit_rows(model.cluster_centers_)
    selected_indices = []
    for label in range(count):
        members = np.flatnonzero(labels == label)
        similarities = candidate_units[members] @ centers[label]
        selected_indices.append(int(members[int(np.argmax(similarities))]))
    return selected_indices


def load_vectors(path: Path) -> dict[str, np.ndarray]:
    """Load finite, nonzero, one-dimensional vectors from a Kaldi archive."""
    vectors: dict[str, np.ndarray] = {}
    for key, value in load_ark(str(path)):
        vector = np.asarray(value, dtype=np.float32).squeeze()
        if (
            vector.ndim == 1
            and vector.size > 0
            and np.isfinite(vector).all()
            and np.any(vector)
        ):
            vectors[str(key)] = vector
    return vectors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--corpus-ark", type=Path, required=True)
    parser.add_argument("--named-ark", type=Path, required=True)
    parser.add_argument("--output-ark", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--common-voice-count", type=int, default=3)
    parser.add_argument("--minimum-duration", type=float, default=2.0)
    parser.add_argument("--maximum-duration", type=float, default=10.0)
    args = parser.parse_args()

    if args.common_voice_count < 1:
        parser.error("--common-voice-count must be positive")

    named = load_vectors(args.named_ark)
    for required in ("lada", "dmytro"):
        if required not in named:
            raise RuntimeError(f"{required!r} is not in {args.named_ark}")

    frame = pd.read_parquet(args.manifest)
    eligible = frame[
        frame["utterance_id"].astype(str).str.startswith("cv22_")
        & frame["duration"].between(args.minimum_duration, args.maximum_duration)
        & frame["qc_flags"].map(len).eq(0)
    ].copy()
    eligible = eligible.sort_values("utterance_id").set_index("utterance_id")

    corpus = load_vectors(args.corpus_ark)
    candidate_ids = [key for key in eligible.index if key in corpus]
    if len(candidate_ids) < args.common_voice_count:
        raise RuntimeError("not enough eligible Common Voice vectors")

    candidate_values = np.stack([corpus[key] for key in candidate_ids])
    chosen_indices = representative_cluster_medoids(
        candidate_values, args.common_voice_count
    )
    chosen_ids = sorted(candidate_ids[index] for index in chosen_indices)

    output_vectors: dict[str, np.ndarray] = {
        "voice_01_lada": named["lada"],
        "voice_02_dmytro_zero_shot": named["dmytro"],
    }
    records = [
        {
            "voice": "voice_01_lada",
            "source": "legacy_named_embedding",
            "source_utterance_id": None,
            "reference_audio": None,
            "training_status": "trained_lada",
        },
        {
            "voice": "voice_02_dmytro_zero_shot",
            "source": "legacy_named_embedding",
            "source_utterance_id": None,
            "reference_audio": None,
            "training_status": "zero_shot_not_in_training_data",
        },
    ]
    for offset, utterance_id in enumerate(chosen_ids, start=3):
        voice = f"voice_{offset:02d}_common_voice"
        output_vectors[voice] = corpus[utterance_id]
        row = eligible.loc[utterance_id]
        records.append(
            {
                "voice": voice,
                "source": str(row["source"]),
                "source_utterance_id": utterance_id,
                "reference_audio": str(Path(row["audio_path"]).resolve()),
                "reference_text": str(row["text_sanitized"]),
                "reference_duration": float(row["duration"]),
                "training_status": "utterance_embedding_in_training_data",
            }
        )

    names = list(output_vectors)
    units = unit_rows(np.stack([output_vectors[name] for name in names]))
    similarities = units @ units.T
    for row_index, record in enumerate(records):
        record["cosine_similarity"] = {
            names[column_index]: round(float(similarities[row_index, column_index]), 6)
            for column_index in range(len(names))
        }

    args.output_ark.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    save_ark(str(args.output_ark), output_vectors)
    report = {
        "status": "PASS",
        "selection": "deterministic_representative_minibatch_kmeans_medoids",
        "eligible_common_voice_vectors": len(candidate_ids),
        "minimum_duration": args.minimum_duration,
        "maximum_duration": args.maximum_duration,
        "voices": records,
    }
    args.output_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
