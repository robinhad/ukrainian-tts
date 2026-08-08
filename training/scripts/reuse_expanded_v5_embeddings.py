#!/usr/bin/env python3
"""Make v5 embedding archives from the compatible v3 and v4 vectors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from kaldiio import ReadHelper, WriteHelper


def read_scp_roots(root: Path) -> dict[str, np.ndarray]:
    vectors: dict[str, np.ndarray] = {}
    for scp in sorted(root.glob("*/xvector.scp")):
        with ReadHelper(f"scp:{scp}") as reader:
            for key, value in reader:
                if key in vectors:
                    raise RuntimeError(f"Duplicate vector ID: {key}")
                vectors[str(key)] = np.asarray(value, dtype=np.float32).squeeze()
    return vectors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hybrid-manifest", type=Path, required=True)
    parser.add_argument("--v3-hybrid-manifest", type=Path, required=True)
    parser.add_argument("--v3-xvector-root", type=Path, required=True)
    parser.add_argument("--v4-hybrid-manifest", type=Path, required=True)
    parser.add_argument("--v4-xvector-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    target = pd.read_parquet(args.hybrid_manifest).sort_values("utterance_id")
    v3 = pd.read_parquet(args.v3_hybrid_manifest).set_index("utterance_id")
    v4 = pd.read_parquet(args.v4_hybrid_manifest).set_index("utterance_id")
    if target["utterance_id"].duplicated().any():
        raise SystemExit("The target manifest has duplicate utterance IDs.")

    v3_vectors = read_scp_roots(args.v3_xvector_root)
    v4_vectors: dict[str, np.ndarray] | None = None
    sources: dict[str, str] = {}
    vectors: dict[str, np.ndarray] = {}
    changed = []
    for row in target.itertuples(index=False):
        identifier = str(row.utterance_id)
        variant = str(row.embedding_audio_variant)
        if identifier not in v3.index:
            raise SystemExit(f"The v3 hybrid manifest has no ID: {identifier}")
        old_variant = str(v3.loc[identifier, "embedding_audio_variant"])
        if old_variant == variant:
            vector = v3_vectors.get(identifier)
            source = "v3"
        else:
            if identifier not in v4.index:
                raise SystemExit(f"The v4 hybrid manifest has no changed ID: {identifier}")
            v4_variant = str(v4.loc[identifier, "embedding_audio_variant"])
            if v4_variant != variant:
                raise SystemExit(
                    f"No compatible vector exists for {identifier}: target={variant}, "
                    f"v3={old_variant}, v4={v4_variant}"
                )
            if v4_vectors is None:
                v4_vectors = read_scp_roots(args.v4_xvector_root)
            vector = v4_vectors.get(identifier)
            source = "v4"
            changed.append(
                {
                    "utterance_id": identifier,
                    "v3_variant": old_variant,
                    "v5_variant": variant,
                    "reused_from": "v4",
                }
            )
        if vector is None:
            raise SystemExit(f"A reusable vector does not exist for {identifier}")
        if vector.shape != (192,) or not np.isfinite(vector).all() or not np.any(vector):
            raise SystemExit(f"The vector is invalid for {identifier}: {vector.shape}")
        vectors[identifier] = vector
        sources[identifier] = source

    archive_report = {}
    for split, split_frame in target.groupby("split", sort=True):
        output = args.output_root / str(split)
        output.mkdir(parents=True, exist_ok=True)
        ark = output / "xvector.ark"
        scp = output / "xvector.scp"
        with WriteHelper(f"ark,scp:{ark},{scp}") as writer:
            for identifier in sorted(split_frame["utterance_id"].astype(str)):
                writer(identifier, vectors[identifier])
        written = sum(1 for line in scp.read_text(encoding="utf-8").splitlines() if line)
        if written != len(split_frame):
            raise SystemExit(f"The archive count is invalid for {split}")
        archive_report[str(split)] = {
            "records": written,
            "ark": str(ark.resolve()),
            "scp": str(scp.resolve()),
        }

    counts = target["embedding_audio_variant"].value_counts().to_dict()
    if counts != {"raw": len(target) // 2, "clean": len(target) // 2}:
        raise SystemExit(f"The target assignment is not exact 50/50: {counts}")
    report = {
        "status": "PASS",
        "records": int(len(target)),
        "dimensions": [192],
        "counts": {key: int(value) for key, value in sorted(counts.items())},
        "reused_vectors": {
            "v3": sum(value == "v3" for value in sources.values()),
            "v4": sum(value == "v4" for value in sources.values()),
        },
        "changed_assignments": changed,
        "archives": archive_report,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
