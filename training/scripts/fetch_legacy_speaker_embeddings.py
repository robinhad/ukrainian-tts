#!/usr/bin/env python3
"""Fetch and validate the pinned legacy Ukrainian speaker embeddings."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path

import numpy as np
from huggingface_hub import snapshot_download
from kaldiio import load_ark


URL = (
    "https://github.com/robinhad/ukrainian-tts/releases/download/"
    "v6.0.0/spk_xvector.ark"
)
SHA256 = "12cf09307b90d0e9745923300d98b7c41ced88229f7905c5a141588fde1d7530"
EXPECTED_KEYS = {"dmytro", "lada", "mykyta", "oleksa", "tetiana"}
EXPECTED_DIMENSION = 192
EMBEDDING_MODEL = "speechbrain/spkrec-ecapa-voxceleb"
EMBEDDING_MODEL_REVISION = "0f99f2d0ebe89ac095bcc5903c4dd8f72b367286"


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate(path: Path) -> dict:
    actual_hash = file_hash(path)
    if actual_hash != SHA256:
        raise RuntimeError(
            f"speaker embedding SHA-256 mismatch: {actual_hash}; expected {SHA256}"
        )
    vectors = {
        key: np.asarray(value).squeeze()
        for key, value in load_ark(str(path))
    }
    if set(vectors) != EXPECTED_KEYS:
        raise RuntimeError(f"unexpected speaker keys: {sorted(vectors)}")
    for key, vector in vectors.items():
        if vector.shape != (EXPECTED_DIMENSION,):
            raise RuntimeError(f"{key}: expected 192 values, got {vector.shape}")
        if not np.isfinite(vector).all() or not np.any(vector):
            raise RuntimeError(f"{key}: invalid speaker embedding")
    return {
        "status": "PASS",
        "source_url": URL,
        "sha256": actual_hash,
        "dimension": EXPECTED_DIMENSION,
        "keys": sorted(vectors),
        "norms": {
            key: float(np.linalg.norm(vector))
            for key, vector in sorted(vectors.items())
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not args.output.is_file() or file_hash(args.output) != SHA256:
        temporary = args.output.with_suffix(args.output.suffix + ".download")
        urllib.request.urlretrieve(URL, temporary)
        if file_hash(temporary) != SHA256:
            temporary.unlink(missing_ok=True)
            raise RuntimeError("downloaded speaker embedding file has a wrong hash")
        temporary.replace(args.output)

    report = validate(args.output)
    snapshot = Path(snapshot_download(
        repo_id=EMBEDDING_MODEL,
        revision=EMBEDDING_MODEL_REVISION,
    ))
    required_model_files = {
        "classifier.ckpt",
        "embedding_model.ckpt",
        "hyperparams.yaml",
        "label_encoder.txt",
        "mean_var_norm_emb.ckpt",
    }
    if not all((args.model_dir / name).is_file() for name in required_model_files):
        args.model_dir.mkdir(parents=True, exist_ok=True)
        for name in required_model_files:
            shutil.copy2(snapshot / name, args.model_dir / name)
    report["embedding_model"] = EMBEDDING_MODEL
    report["embedding_model_revision"] = EMBEDDING_MODEL_REVISION
    report["embedding_model_dir"] = str(args.model_dir.resolve())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
