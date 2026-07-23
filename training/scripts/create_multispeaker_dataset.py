#!/usr/bin/env python3
"""Build a pinned Common Voice and Lada dataset for speaker-embedding JETS."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Iterable

import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

from training.frontend.sanitize import sanitize_text


CV_DATASET = "speech-uk/cv22-opus"
CV_REVISION = "46e5072c44cdd6d7d371adeba29f1ee7678aaf91"
CV_LICENSE = "CC0-1.0"
EMBEDDING_MODEL = (
    "speechbrain/spkrec-ecapa-voxceleb"
    "@0f99f2d0ebe89ac095bcc5903c4dd8f72b367286"
)
SEED = "777"


def stable_score(value: str) -> str:
    return hashlib.sha256(f"{SEED}\0{value}".encode("utf-8")).hexdigest()


def split_for_text(text_hash: str) -> str:
    """Keep every recording of one sanitized text in one split."""
    bucket = int(stable_score(text_hash)[:8], 16) % 10_000
    if bucket < 200:
        return "eval"
    if bucket < 400:
        return "dev"
    return "train"


def split_name(mode: str, split: str) -> str:
    prefix = "multispeaker_smoke_" if mode == "smoke" else "multispeaker_"
    return prefix + split


def cv_files(limit: int | None = None) -> list[str]:
    api = HfApi()
    names = [
        item.path
        for item in api.list_repo_tree(
            CV_DATASET,
            repo_type="dataset",
            revision=CV_REVISION,
            recursive=True,
        )
        if item.path.endswith(".parquet")
    ]
    names.sort(key=lambda name: int(Path(name).stem))
    return names if limit is None else names[:limit]


def iter_cv_rows(files: Iterable[str]) -> Iterable[dict]:
    for name in files:
        path = hf_hub_download(
            repo_id=CV_DATASET,
            repo_type="dataset",
            revision=CV_REVISION,
            filename=name,
        )
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(
            batch_size=256,
            columns=["audio", "duration", "transcription"],
        ):
            yield from batch.to_pylist()


def materialize_cv(
    *,
    output_root: Path,
    files: list[str],
    min_duration: float,
    max_duration: float,
) -> tuple[list[dict], Counter]:
    raw_dir = output_root / "raw" / "common_voice"
    raw_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    excluded: Counter = Counter()
    seen_audio: set[str] = set()
    for source in iter_cv_rows(files):
        text = str(source.get("transcription") or "").strip()
        duration = float(source.get("duration") or 0.0)
        audio = source.get("audio") or {}
        content = audio.get("bytes")
        source_path = str(audio.get("path") or "audio.opus")
        if not text or content is None:
            excluded["missing_pair"] += 1
            continue
        if not min_duration <= duration <= max_duration:
            excluded["duration"] += 1
            continue
        sanitized = sanitize_text(text)
        if not sanitized:
            excluded["empty_sanitized_text"] += 1
            continue
        if not any(character.isalpha() for character in sanitized):
            excluded["no_spoken_text"] += 1
            continue
        audio_hash = hashlib.sha256(content).hexdigest()
        if audio_hash in seen_audio:
            excluded["duplicate_audio"] += 1
            continue
        seen_audio.add(audio_hash)
        text_hash = hashlib.sha256(sanitized.encode("utf-8")).hexdigest()
        identifier = f"cv22_{Path(source_path).stem}_{audio_hash[:10]}"
        extension = Path(source_path).suffix or ".opus"
        target = raw_dir / f"{identifier}{extension}"
        target.write_bytes(content)
        rows.append(
            {
                "utterance_id": identifier,
                # cv22-opus does not expose client_id. An utterance-level embedding
                # is the speaker condition. A per-utterance label prevents false
                # claims that unrelated clips have one speaker identity.
                "speaker_id": identifier,
                "audio_path": str(target.resolve()),
                "text_raw": text,
                "duration_source": duration,
                "source": f"hf://datasets/{CV_DATASET}@{CV_REVISION}",
                "source_license": CV_LICENSE,
                "source_audio_path": source_path,
                "source_group": f"cv_text_{text_hash}",
                "split": split_for_text(text_hash),
                "audio_sha256_source": audio_hash,
                "text_sha256_source": text_hash,
                "speaker_embedding_mode": "utterance",
                "speaker_embedding_model": EMBEDDING_MODEL,
                "speaker_identity_status": "anonymous_no_client_id",
            }
        )
    return rows, excluded


def load_lada(path: Path) -> list[dict]:
    frame = pd.read_parquet(path)
    rows = []
    for source in frame.to_dict("records"):
        rows.append(
            {
                "utterance_id": str(source["utterance_id"]),
                "speaker_id": "lada",
                "audio_path": str(source["audio_path"]),
                "text_raw": str(source["text_raw"]),
                "duration_source": float(source["duration"]),
                "source": str(source["source"]),
                "source_license": str(source["source_license"]),
                "source_audio_path": str(
                    source.get("source_audio_path") or source["audio_path"]
                ),
                "source_group": str(source.get("source_group") or "lada"),
                "split": str(source["split"]),
                "audio_sha256_source": str(source["audio_sha256"]),
                "text_sha256_source": str(source["text_sha256"]),
                "speaker_embedding_mode": "utterance",
                "speaker_embedding_model": EMBEDDING_MODEL,
                "speaker_identity_status": "explicit_lada",
            }
        )
    return rows


def load_dmytro(path: Path | None) -> list[dict]:
    if path is None:
        return []
    frame = pd.read_parquet(path)
    required = {"utterance_id", "audio_path", "text_raw", "duration", "split"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"Dmytro manifest is missing columns: {missing}")
    rows = []
    for source in frame.to_dict("records"):
        text = str(source["text_raw"])
        sanitized = sanitize_text(text)
        rows.append(
            {
                "utterance_id": str(source["utterance_id"]),
                "speaker_id": "dmytro",
                "audio_path": str(source["audio_path"]),
                "text_raw": text,
                "duration_source": float(source["duration"]),
                "source": str(source.get("source") or "dmytro_manifest"),
                "source_license": str(source.get("source_license") or "UNKNOWN"),
                "source_audio_path": str(
                    source.get("source_audio_path") or source["audio_path"]
                ),
                "source_group": str(source.get("source_group") or "dmytro"),
                "split": str(source["split"]),
                "audio_sha256_source": str(
                    source.get("audio_sha256") or source.get("audio_sha256_source") or ""
                ),
                "text_sha256_source": hashlib.sha256(
                    sanitized.encode("utf-8")
                ).hexdigest(),
                "speaker_embedding_mode": "utterance",
                "speaker_embedding_model": EMBEDDING_MODEL,
                "speaker_identity_status": "explicit_dmytro",
            }
        )
    return rows


def remove_cross_source_text_overlap(
    cv_rows: list[dict], explicit_speaker_rows: list[dict]
) -> tuple[list[dict], int]:
    """Prefer explicit speaker data when sanitized text occurs in both sources."""
    explicit_hashes = {
        hashlib.sha256(
            sanitize_text(str(row["text_raw"])).encode("utf-8")
        ).hexdigest()
        for row in explicit_speaker_rows
    }
    filtered = [
        row for row in cv_rows if str(row["text_sha256_source"]) not in explicit_hashes
    ]
    return filtered, len(cv_rows) - len(filtered)


def select_smoke(rows: list[dict], per_source: dict[str, dict[str, int]]) -> list[dict]:
    selected: list[dict] = []
    for source_name, limits in per_source.items():
        source_rows = [
            row
            for row in rows
            if ("cv22" if row["utterance_id"].startswith("cv22_") else row["speaker_id"])
            == source_name
        ]
        for split, count in limits.items():
            candidates = [row for row in source_rows if row["split"] == split]
            candidates.sort(key=lambda row: stable_score(row["utterance_id"]))
            if len(candidates) < count:
                raise RuntimeError(
                    f"{source_name}/{split}: found {len(candidates)}, need {count}"
                )
            selected.extend(candidates[:count])
    return selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--lada-manifest",
        type=Path,
        default=Path("training/data/full_trimmed/manifests/all.parquet"),
    )
    parser.add_argument("--dmytro-manifest", type=Path)
    parser.add_argument("--mode", choices=["smoke", "full"], required=True)
    parser.add_argument("--cv-shards", type=int)
    parser.add_argument("--min-duration", type=float, default=2.0)
    parser.add_argument("--max-duration", type=float, default=12.0)
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    shard_limit = args.cv_shards
    if shard_limit is None and args.mode == "smoke":
        shard_limit = 4
    files = cv_files(shard_limit)
    cv_rows, excluded = materialize_cv(
        output_root=args.output_root,
        files=files,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
    )
    lada_rows = load_lada(args.lada_manifest)
    dmytro_rows = load_dmytro(args.dmytro_manifest)
    cv_rows, overlap_count = remove_cross_source_text_overlap(
        cv_rows, lada_rows + dmytro_rows
    )
    if overlap_count:
        excluded["cross_source_duplicate_text"] += overlap_count
    combined = cv_rows + lada_rows + dmytro_rows

    if args.mode == "smoke":
        limits = {
            "cv22": {"train": 128, "dev": 16, "eval": 16},
            "lada": {"train": 128, "dev": 16, "eval": 16},
        }
        if dmytro_rows:
            limits["dmytro"] = {"train": 64, "dev": 8, "eval": 8}
        combined = select_smoke(combined, limits)

    for row in combined:
        row["split"] = split_name(args.mode, str(row["split"]))
    combined.sort(key=lambda row: row["utterance_id"])
    output = args.output_root / "source_records.jsonl"
    with output.open("w", encoding="utf-8") as stream:
        for row in combined:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    source_counts = Counter(
        "common_voice"
        if row["utterance_id"].startswith("cv22_")
        else str(row["speaker_id"])
        for row in combined
    )
    split_counts = Counter(str(row["split"]) for row in combined)
    report = {
        "status": "PASS",
        "mode": args.mode,
        "artifact": str(output),
        "common_voice_dataset": CV_DATASET,
        "common_voice_revision": CV_REVISION,
        "common_voice_shards": files,
        "speaker_embedding_model": EMBEDDING_MODEL,
        "source_counts": dict(sorted(source_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "dmytro_training_rows": source_counts.get("dmytro", 0),
        "dmytro_inference_status": "legacy_192d_embedding_available",
        "excluded_common_voice": dict(sorted(excluded.items())),
    }
    report_path = args.output_root / "source_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
