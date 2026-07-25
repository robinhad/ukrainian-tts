#!/usr/bin/env python3
"""Select five deterministic speaker vectors from the expanded-v3 corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from kaldiio import save_ark

from training.scripts.select_diverse_listening_voices import (
    load_vectors,
    representative_cluster_medoids,
    unit_rows,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--corpus-ark", type=Path, required=True)
    parser.add_argument("--output-ark", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()

    frame = pd.read_parquet(args.manifest).sort_values("utterance_id")
    frame = frame[
        frame["duration"].between(2.0, 10.0)
        & frame["qc_flags"].map(len).eq(0)
    ].set_index("utterance_id")
    corpus = load_vectors(args.corpus_ark)
    identifiers = [str(value) for value in frame.index if str(value) in corpus]
    if len(identifiers) < 5:
        raise RuntimeError("The corpus has fewer than five eligible speaker vectors.")
    values = np.stack([corpus[key] for key in identifiers])
    selected = sorted(
        identifiers[index] for index in representative_cluster_medoids(values, 5)
    )
    voices = {
        f"voice_{index:02d}": corpus[identifier]
        for index, identifier in enumerate(selected, 1)
    }
    units = unit_rows(np.stack(list(voices.values())))
    similarity = units @ units.T
    records = []
    names = list(voices)
    for row_index, (voice, identifier) in enumerate(zip(names, selected)):
        row = frame.loc[identifier]
        records.append(
            {
                "voice": voice,
                "source_utterance_id": identifier,
                "source_id": str(row["source_id"]),
                "reference_audio": str(Path(row["audio_path"]).resolve()),
                "reference_text": str(row["text_sanitized"]),
                "cosine_similarity": {
                    name: round(float(similarity[row_index, column]), 6)
                    for column, name in enumerate(names)
                },
            }
        )
    args.output_ark.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    save_ark(str(args.output_ark), voices)
    args.output_report.write_text(
        json.dumps(
            {
                "status": "PASS",
                "selection": "deterministic_representative_minibatch_kmeans_medoids",
                "voices": records,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
