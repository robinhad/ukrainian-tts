#!/usr/bin/env python3
"""Materialize one permitted Hugging Face Parquet source in small shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable

import fsspec
import pyarrow.parquet as pq
import requests
import soundfile as sf
from huggingface_hub import HfApi, hf_hub_download

from training.frontend.sanitize import sanitize_text
from training.scripts.source_policy import (
    check_disk,
    load_registry,
    read_hf_token,
    validate_record,
)


TEXT_FIELDS = ("transcription", "transcribed_text", "text", "raw_transcription")
SPEAKER_FIELDS = ("speaker_id", "client_id", "speaker", "speaker_name")
GROUP_FIELDS = ("document_id", "book_id", "article_id", "part", "segment")
KNOWN_SPEAKERS = {
    "opentts_lada": "lada",
    "opentts_tetiana": "tetiana",
    "opentts_mykyta": "mykyta",
}


def stable_id(source_id: str, source_path: str, audio_hash: str) -> str:
    stem = Path(source_path).stem.replace(" ", "_")[:64] or "audio"
    return f"{source_id}_{stem}_{audio_hash[:12]}"


def first_text(row: dict[str, Any]) -> str:
    for field in TEXT_FIELDS:
        value = row.get(field)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def first_value(row: dict[str, Any], fields: tuple[str, ...]) -> str | None:
    for field in fields:
        value = row.get(field)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def parquet_files(
    repo_id: str,
    revision: str,
    token: str,
    maximum_shards: int | None,
    subset: str | None = None,
) -> list[str]:
    api = HfApi(token=token)
    files = sorted(
        item.path
        for item in api.list_repo_tree(
            repo_id,
            repo_type="dataset",
            revision=revision,
            recursive=True,
        )
        if item.path.endswith(".parquet")
        and (subset is None or f"/{subset}/" in f"/{item.path}")
    )
    if not files:
        response = requests.get(
            "https://datasets-server.huggingface.co/parquet",
            params={"dataset": repo_id},
            headers={"Authorization": f"Bearer {token}"},
            timeout=60,
        )
        if response.ok:
            files = [
                item["url"]
                for item in response.json().get("parquet_files", [])
                if item.get("url") and (subset is None or item.get("config") == subset)
            ]
    if maximum_shards is not None:
        files = files[:maximum_shards]
    return files


def source_split(path: str) -> str:
    name = Path(path).name.lower()
    for split in ("validation", "dev", "test", "train"):
        if name.startswith(split + "-"):
            return split
    return "train"


def iter_rows(path: Path | str, token: str) -> Iterable[dict[str, Any]]:
    if isinstance(path, str) and path.startswith(("https://", "http://")):
        handle = fsspec.open(
            path,
            mode="rb",
            headers={"Authorization": f"Bearer {token}"},
            block_size=8 * 1024 * 1024,
            cache_type="readahead",
        ).open()
        parquet = pq.ParquetFile(handle)
    else:
        handle = None
        parquet = pq.ParquetFile(path)
    available = set(parquet.schema_arrow.names)
    requested = {
        "audio",
        "duration",
        *TEXT_FIELDS,
        *SPEAKER_FIELDS,
        *GROUP_FIELDS,
    }
    columns = sorted(available & requested)
    if "audio" not in columns:
        raise RuntimeError(f"The Parquet shard has no audio column: {path}")
    try:
        for batch in parquet.iter_batches(batch_size=64, columns=columns):
            yield from batch.to_pylist()
    finally:
        if handle is not None:
            handle.close()


def canonicalize_audio(content: bytes, source_path: str, target: Path) -> None:
    suffix = Path(source_path).suffix or ".audio"
    with tempfile.TemporaryDirectory(prefix="uktts-hf-audio-") as temp_dir:
        source = Path(temp_dir) / f"source{suffix}"
        source.write_bytes(content)
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(source),
                "-map_metadata",
                "-1",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "flac",
                str(target),
            ],
            check=True,
        )


def materialize_row(
    row: dict[str, Any],
    *,
    source_id: str,
    source: dict[str, Any],
    raw_dir: Path,
    shard_name: str,
) -> tuple[dict[str, Any] | None, str | None]:
    text = first_text(row)
    sanitized = sanitize_text(text)
    audio = row.get("audio") or {}
    if not isinstance(audio, dict):
        return None, "invalid_audio_object"
    content = audio.get("bytes")
    source_path = str(audio.get("path") or "audio.bin")
    if not content or not sanitized:
        return None, "missing_audio_or_text"
    if not any(character.isalpha() for character in sanitized):
        return None, "no_spoken_text"
    audio_hash = hashlib.sha256(content).hexdigest()
    utterance_id = stable_id(source_id, source_path, audio_hash)
    target = raw_dir / f"{utterance_id}.flac"
    if not target.is_file():
        canonicalize_audio(content, source_path, target)
    info = sf.info(target)
    duration = float(info.duration)
    if not 2.0 <= duration <= 12.0:
        target.unlink(missing_ok=True)
        return None, "duration"
    speaker = first_value(row, SPEAKER_FIELDS) or KNOWN_SPEAKERS.get(source_id)
    if speaker is None:
        speaker = utterance_id
        speaker_status = "anonymous_utterance_condition"
        kaldi_speaker = utterance_id
        speaker_stratum = utterance_id
    elif source_id in KNOWN_SPEAKERS:
        kaldi_speaker = source_id
        speaker_stratum = source_id
        speaker_status = f"explicit_{speaker}"
    else:
        speaker_status = "source_speaker_id"
        kaldi_speaker = utterance_id
        speaker_stratum = f"{source_id}:{speaker}"
    group = first_value(row, GROUP_FIELDS)
    if not group:
        stem = Path(source_path).stem
        if stem.isdigit():
            group = f"{source_id}:sequential-{int(stem) // 50:08d}"
        else:
            group = f"{source_id}:{shard_name}:{speaker}"
    record = {
        "utterance_id": utterance_id,
        "speaker_id": kaldi_speaker,
        "speaker_stratum_id": speaker_stratum,
        "audio_path": str(target.resolve()),
        "canonical_raw_audio_path": str(target.resolve()),
        "text_raw": text,
        "duration_source": float(row.get("duration") or duration),
        "duration": duration,
        "sample_rate": int(info.samplerate),
        "source_id": source_id,
        "source": f"hf://datasets/{source['repo_id']}@{source['revision']}",
        "source_license": source["license"],
        "source_audio_path": source_path,
        "source_group": str(group),
        "source_original_split": source_split(shard_name),
        "audio_sha256_source": audio_hash,
        "text_sha256_source": hashlib.sha256(
            sanitized.encode("utf-8")
        ).hexdigest(),
        "speaker_embedding_mode": "utterance",
        "speaker_embedding_model": (
            "speechbrain/spkrec-ecapa-voxceleb"
            "@0f99f2d0ebe89ac095bcc5903c4dd8f72b367286"
        ),
        "speaker_identity_status": speaker_status,
        "label_kind": "human",
        "license_evidence": source.get("evidence"),
    }
    errors = validate_record(record, {"sources": {source_id: source}})
    if errors:
        target.unlink(missing_ok=True)
        return None, "policy:" + "|".join(errors)
    return record, None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--maximum-shards", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    registry = load_registry(args.registry)
    source = registry["sources"].get(args.source_id)
    if not source or source.get("connector") != "huggingface":
        parser.error("The source must be an enabled Hugging Face source.")
    if not source.get("enabled") or source.get("decision") != "allow":
        parser.error("The source policy does not allow this source.")
    if source.get("kind") != "labeled":
        parser.error("Use the unlabeled pipeline for this source.")

    args.output_root.mkdir(parents=True, exist_ok=True)
    disk = check_disk(args.output_root.parent, registry)
    if disk["status"] == "STOP":
        raise SystemExit("Free disk space is below the 60 GiB stop threshold.")
    token = read_hf_token()
    files = parquet_files(
        source["repo_id"],
        source["revision"],
        token,
        args.maximum_shards,
        source.get("subset"),
    )
    if not files:
        raise SystemExit("The source has no repository Parquet shards.")
    source_root = args.output_root / args.source_id
    raw_dir = source_root / "canonical_raw_16k"
    records_path = source_root / "records.jsonl"
    prior: dict[str, dict[str, Any]] = {}
    if args.resume and records_path.is_file():
        prior = {
            row["utterance_id"]: row
            for row in (
                json.loads(line)
                for line in records_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        }
    records = dict(prior)
    excluded: dict[str, int] = {}
    selected_new = 0
    shard_reports = []
    for filename in files:
        disk = check_disk(args.output_root.parent, registry)
        if disk["status"] == "STOP":
            raise SystemExit("Free disk space is below the 60 GiB stop threshold.")
        if filename.startswith(("https://", "http://")):
            shard: Path | str = filename
        else:
            shard = Path(
                hf_hub_download(
                    repo_id=source["repo_id"],
                    repo_type="dataset",
                    revision=source["revision"],
                    filename=filename,
                    token=token,
                )
            )
        shard_selected = 0
        for row in iter_rows(shard, token):
            record, reason = materialize_row(
                row,
                source_id=args.source_id,
                source=source,
                raw_dir=raw_dir,
                shard_name=filename,
            )
            if record is None:
                excluded[reason or "unknown"] = excluded.get(reason or "unknown", 0) + 1
                continue
            records[record["utterance_id"]] = record
            selected_new += 1
            shard_selected += 1
            if args.limit is not None and selected_new >= args.limit:
                break
        shard_reports.append({"file": filename, "selected": shard_selected})
        records_path.parent.mkdir(parents=True, exist_ok=True)
        records_path.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                for row in sorted(records.values(), key=lambda item: item["utterance_id"])
            ),
            encoding="utf-8",
        )
        if args.limit is not None and selected_new >= args.limit:
            break
    report = {
        "status": "PASS",
        "source_id": args.source_id,
        "records": len(records),
        "selected_new": selected_new,
        "excluded": dict(sorted(excluded.items())),
        "shards": shard_reports,
        "artifact": str(records_path),
        "disk": check_disk(args.output_root.parent, registry),
    }
    report_path = source_root / "ingest_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
