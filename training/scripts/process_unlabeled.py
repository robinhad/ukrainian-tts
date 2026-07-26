#!/usr/bin/env python3
"""Diarize unlabeled audio and accept high-confidence Ukrainian transcripts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from huggingface_hub import snapshot_download

from training.frontend.sanitize import sanitize_text
from training.scripts.source_policy import load_registry, read_hf_token


UKRAINIAN_ALPHABET = set("абвгґдеєжзиіїйклмнопрстуфхцчшщьюя")


def parse_segment(value: Any) -> tuple[float, float, int]:
    if isinstance(value, str):
        parts = value.replace(",", " ").split()
    elif isinstance(value, (tuple, list)):
        parts = list(value)
    else:
        raise ValueError(f"Unsupported diarization segment: {value!r}")
    if len(parts) != 3:
        raise ValueError(f"Expected start, end, and speaker: {value!r}")
    start, end = float(parts[0]), float(parts[1])
    speaker_text = str(parts[2])
    speaker_digits = "".join(character for character in speaker_text if character.isdigit())
    speaker = int(speaker_digits or speaker_text)
    if not 0 <= start < end:
        raise ValueError(f"Invalid diarization interval: {value!r}")
    return start, end, speaker


def non_overlapping_segments(
    segments: list[tuple[float, float, int]],
) -> list[tuple[float, float, int]]:
    """Return intervals that have exactly one active speaker."""
    boundaries = sorted({point for start, end, _ in segments for point in (start, end)})
    pieces: list[tuple[float, float, int]] = []
    for left, right in zip(boundaries, boundaries[1:]):
        active = {
            speaker
            for start, end, speaker in segments
            if start < right and end > left
        }
        if len(active) == 1 and right > left:
            speaker = next(iter(active))
            if pieces and pieces[-1][2] == speaker and math.isclose(pieces[-1][1], left):
                pieces[-1] = (pieces[-1][0], right, speaker)
            else:
                pieces.append((left, right, speaker))
    return pieces


def low_energy_split(
    audio: np.ndarray,
    sample_rate: int,
    start: float,
    end: float,
    maximum_seconds: float = 12.0,
) -> list[tuple[float, float]]:
    """Split a long interval at a low-energy point near each target boundary."""
    result = []
    cursor = start
    while end - cursor > maximum_seconds:
        target = cursor + maximum_seconds
        search_start = max(cursor + 2.0, target - 1.0)
        search_end = min(end - 2.0, target + 1.0)
        if search_end <= search_start:
            cut = target
        else:
            frame = max(1, round(sample_rate * 0.04))
            first = max(0, round(search_start * sample_rate))
            last = min(len(audio) - frame, round(search_end * sample_rate))
            starts = np.arange(first, last + 1, frame, dtype=np.int64)
            energy = np.array(
                [
                    float(np.mean(np.square(audio[index : index + frame])))
                    for index in starts
                ]
            )
            cut = float(starts[int(np.argmin(energy))]) / sample_rate
        result.append((cursor, cut))
        cursor = cut
    result.append((cursor, end))
    return result


def diarization_chunks(
    audio: np.ndarray,
    sample_rate: int,
    maximum_seconds: float,
) -> list[tuple[int, float, np.ndarray]]:
    """Split source audio before feature extraction to limit GPU memory."""
    if sample_rate <= 0:
        raise ValueError("The sample rate must be positive.")
    if maximum_seconds <= 0:
        raise ValueError("The maximum chunk duration must be positive.")
    maximum_samples = max(1, round(sample_rate * maximum_seconds))
    return [
        (index, first / sample_rate, audio[first : first + maximum_samples])
        for index, first in enumerate(range(0, len(audio), maximum_samples))
    ]


def diarize_in_chunks(
    diar_model: Any,
    audio: np.ndarray,
    sample_rate: int,
    maximum_seconds: float,
    maximum_speakers: int,
) -> list[tuple[float, float, int]]:
    """Diarize bounded chunks and return source-relative time intervals."""
    if maximum_speakers < 1:
        raise ValueError("The maximum speaker count must be positive.")
    segments: list[tuple[float, float, int]] = []
    for chunk_index, offset, chunk in diarization_chunks(
        audio,
        sample_rate,
        maximum_seconds,
    ):
        predicted = diar_model.diarize(
            audio=[chunk],
            batch_size=1,
            sample_rate=sample_rate,
        )[0]
        for value in predicted:
            start, end, speaker = parse_segment(value)
            segments.append(
                (
                    start + offset,
                    end + offset,
                    chunk_index * maximum_speakers + speaker,
                )
            )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return segments


def source_progress_key(source: dict[str, Any]) -> str:
    material = (
        f"{source['audio_sha256_source']}:{source.get('source_group', '')}"
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        for row in rows:
            stream.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            )
        stream.flush()
        os.fsync(stream.fileno())


def load_progress(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    entries: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        1,
    ):
        if not line.strip():
            continue
        entry = json.loads(line)
        key = str(entry.get("source_key") or "")
        if not key:
            raise ValueError(
                f"Progress line {line_number} has no source key: {path}"
            )
        entries[key] = entry
    return entries


def ukrainian_letter_ratio(text: str) -> float:
    letters = [character.lower() for character in text if character.isalpha()]
    if not letters:
        return 0.0
    return sum(character in UKRAINIAN_ALPHABET for character in letters) / len(letters)


def hypothesis_language(hypothesis: Any) -> str | None:
    for name in ("lang", "language", "language_id"):
        value = getattr(hypothesis, name, None)
        if value:
            return str(value)
    values = getattr(hypothesis, "langs", None)
    if values:
        if isinstance(values, str):
            return values
        counts: dict[str, int] = {}
        for value in values:
            key = str(value)
            counts[key] = counts.get(key, 0) + 1
        return max(counts, key=counts.get)
    return None


def hypothesis_confidence(hypothesis: Any) -> float | None:
    for name in ("confidence", "mean_confidence"):
        value = getattr(hypothesis, name, None)
        if value is not None:
            return float(value)
    for name in ("word_confidence", "token_confidence"):
        values = getattr(hypothesis, name, None)
        if values is not None and len(values):
            return float(np.mean(np.asarray(values, dtype=np.float64)))
    return None


def enable_asr_confidence(asr_model: Any) -> None:
    """Enable token confidence without importing optional NeMo CLI helpers."""
    from nemo.collections.asr.parts.utils.asr_confidence_utils import (
        ConfidenceConfig,
    )
    from omegaconf import OmegaConf, open_dict

    decoding_cfg = asr_model.cfg.decoding
    with open_dict(decoding_cfg):
        if "confidence_cfg" not in decoding_cfg:
            decoding_cfg.confidence_cfg = OmegaConf.structured(
                ConfidenceConfig(aggregation="mean")
            )
        decoding_cfg.confidence_cfg.preserve_frame_confidence = True
        decoding_cfg.confidence_cfg.preserve_token_confidence = True
        decoding_cfg.confidence_cfg.preserve_word_confidence = False
        strategy = decoding_cfg.get("strategy", "greedy")
        if strategy in ("greedy", "greedy_batch"):
            decoding_cfg.greedy.preserve_frame_confidence = True
        elif strategy in ("malsd_batch", "maes_batch"):
            decoding_cfg.beam.preserve_frame_confidence = True
    asr_model.change_decoding_strategy(decoding_cfg, verbose=False)


def load_models(registry: dict[str, Any], cache: Path) -> tuple[Any, Any]:
    from nemo.collections.asr.models import ASRModel, SortformerEncLabelModel

    token = read_hf_token()
    diar = registry["models"]["diarizer"]
    asr = registry["models"]["asr"]
    diar_dir = Path(
        snapshot_download(
            diar["repo_id"],
            revision=diar["revision"],
            token=token,
            cache_dir=cache,
            allow_patterns=["*.nemo"],
        )
    )
    asr_dir = Path(
        snapshot_download(
            asr["repo_id"],
            revision=asr["revision"],
            token=token,
            cache_dir=cache,
            allow_patterns=["*.nemo"],
        )
    )
    diar_model = SortformerEncLabelModel.restore_from(
        str(next(diar_dir.glob("*.nemo"))), map_location="cuda", strict=False
    )
    asr_model = ASRModel.restore_from(
        str(next(asr_dir.glob("*.nemo"))), map_location="cuda", strict=False
    )
    enable_asr_confidence(asr_model)
    diar_model.eval()
    asr_model.eval()
    diar_model.sortformer_modules.chunk_len = 340
    diar_model.sortformer_modules.chunk_right_context = 40
    diar_model.sortformer_modules.fifo_len = 40
    diar_model.sortformer_modules.spkcache_update_period = 300
    diar_model.sortformer_modules.spkcache_len = 188
    diar_model.sortformer_modules._check_streaming_parameters()
    return diar_model, asr_model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--records", type=Path, nargs="+", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-records", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--model-cache", type=Path, required=True)
    parser.add_argument("--source-start", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--delete-source-audio", action="store_true")
    args = parser.parse_args()
    if args.source_start < 0:
        parser.error("--source-start must not be negative.")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive.")

    registry = load_registry(args.registry)
    diar_cfg = registry["models"]["diarizer"]
    asr_cfg = registry["models"]["asr"]
    sources = [
        json.loads(line)
        for path in args.records
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    sources = sources[args.source_start :]
    if args.limit is not None:
        sources = sources[: args.limit]
    if not sources:
        raise SystemExit("No source records are selected.")

    if args.report:
        progress_path = args.report.with_name(
            f"{args.report.stem}.progress.jsonl"
        )
    else:
        progress_path = args.output_records.with_name(
            f"{args.output_records.name}.progress.jsonl"
        )
    selected_keys = {source_progress_key(source) for source in sources}
    progress = {
        key: entry
        for key, entry in load_progress(progress_path).items()
        if key in selected_keys
    }
    resumed_source_files = len(progress)
    accepted_count = sum(
        int(entry.get("accepted_segments", 0))
        for entry in progress.values()
    )
    rejected: dict[str, int] = {}
    for entry in progress.values():
        for reason, count in entry.get("rejected", {}).items():
            rejected[reason] = rejected.get(reason, 0) + int(count)

    args.output_records.parent.mkdir(parents=True, exist_ok=True)
    if not args.append and not progress:
        args.output_records.write_text("", encoding="utf-8")
    elif not args.output_records.exists():
        args.output_records.touch()

    diar_model, asr_model = load_models(registry, args.model_cache)
    for source in sources:
        progress_key = source_progress_key(source)
        if progress_key in progress:
            continue
        source_accepted: list[dict[str, Any]] = []
        source_rejected: dict[str, int] = {}
        audio, sample_rate = sf.read(
            source["audio_path"], always_2d=True, dtype="float32"
        )
        mono = np.mean(audio, axis=1)
        if sample_rate != 16000:
            with tempfile.TemporaryDirectory(prefix="uktts-unlabeled-") as temp_dir:
                converted = Path(temp_dir) / "source.wav"
                subprocess.run(
                    [
                        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
                        "-y", "-i", source["audio_path"], "-ac", "1", "-ar", "16000",
                        "-c:a", "pcm_f32le", str(converted),
                    ],
                    check=True,
                )
                mono, sample_rate = sf.read(converted, dtype="float32")
        parsed = diarize_in_chunks(
            diar_model,
            mono,
            sample_rate,
            float(diar_cfg["maximum_chunk_seconds"]),
            int(diar_cfg["maximum_speakers"]),
        )
        single = non_overlapping_segments(parsed)
        for start, end, speaker in single:
            for part_start, part_end in low_energy_split(
                mono, sample_rate, start, end
            ):
                duration = part_end - part_start
                if not 2.0 <= duration <= 12.0:
                    source_rejected["duration"] = (
                        source_rejected.get("duration", 0) + 1
                    )
                    continue
                samples = mono[
                    round(part_start * sample_rate) : round(part_end * sample_rate)
                ]
                segment_key = (
                    f"{source['audio_sha256_source']}:{source['source_group']}:"
                    f"{part_start:.3f}:{part_end:.3f}:{speaker}"
                )
                identifier = (
                    f"{source['source_id']}_pseudo_"
                    f"{hashlib.sha256(segment_key.encode()).hexdigest()[:20]}"
                )
                target = args.output_root / f"{identifier}.flac"
                target.parent.mkdir(parents=True, exist_ok=True)
                sf.write(target, samples, sample_rate, subtype="PCM_16")
                hypothesis = asr_model.transcribe(
                    [str(target)], timestamps=True, return_hypotheses=True
                )[0]
                text = sanitize_text(str(hypothesis.text))
                language = hypothesis_language(hypothesis)
                confidence = hypothesis_confidence(hypothesis)
                script_ratio = ukrainian_letter_ratio(text)
                if (
                    language is not None
                    and language != asr_cfg["accepted_language"]
                ):
                    source_rejected["language"] = (
                        source_rejected.get("language", 0) + 1
                    )
                    target.unlink(missing_ok=True)
                    continue
                if confidence is None or confidence < float(
                    asr_cfg["minimum_mean_confidence"]
                ):
                    source_rejected["confidence"] = (
                        source_rejected.get("confidence", 0) + 1
                    )
                    target.unlink(missing_ok=True)
                    continue
                if script_ratio < float(asr_cfg["minimum_ukrainian_letter_ratio"]):
                    source_rejected["script_ratio"] = (
                        source_rejected.get("script_ratio", 0) + 1
                    )
                    target.unlink(missing_ok=True)
                    continue
                accepted_language = language or "uk_script_inferred"
                audio_hash = hashlib.sha256(target.read_bytes()).hexdigest()
                source_scope = source["audio_sha256_source"][:20]
                source_accepted.append(
                    {
                        **source,
                        "utterance_id": identifier,
                        "speaker_id": (
                            f"{source['source_id']}:{source_scope}:"
                            f"speaker-{speaker}"
                        ),
                        "audio_path": str(target.resolve()),
                        "canonical_raw_audio_path": str(target.resolve()),
                        "text_raw": text,
                        "duration_source": duration,
                        "duration": duration,
                        "sample_rate": sample_rate,
                        "source_group": (
                            f"{source['source_group']}:source-{source_scope}:"
                            f"speaker-{speaker}"
                        ),
                        "parent_audio_sha256_source": (
                            source["audio_sha256_source"]
                        ),
                        "audio_sha256_source": audio_hash,
                        "text_sha256_source": hashlib.sha256(
                            text.encode("utf-8")
                        ).hexdigest(),
                        "speaker_embedding_mode": "utterance",
                        "speaker_embedding_model": (
                            "speechbrain/spkrec-ecapa-voxceleb"
                            "@0f99f2d0ebe89ac095bcc5903c4dd8f72b367286"
                        ),
                        "speaker_identity_status": "diarized_local_speaker",
                        "label_kind": "pseudo",
                        "diarization_model": (
                            f"{diar_cfg['repo_id']}@{diar_cfg['revision']}"
                        ),
                        "asr_model": f"{asr_cfg['repo_id']}@{asr_cfg['revision']}",
                        "asr_language": accepted_language,
                        "asr_mean_confidence": confidence,
                    }
                )
        append_jsonl(args.output_records, source_accepted)
        progress_entry = {
            "source_key": progress_key,
            "audio_sha256_source": source["audio_sha256_source"],
            "source_group": source.get("source_group"),
            "accepted_segments": len(source_accepted),
            "rejected": dict(sorted(source_rejected.items())),
        }
        append_jsonl(progress_path, [progress_entry])
        progress[progress_key] = progress_entry
        accepted_count += len(source_accepted)
        for reason, count in source_rejected.items():
            rejected[reason] = rejected.get(reason, 0) + count
        del audio, mono
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    cleanup_trigger_gib = float(registry["policy"]["free_disk_stop_gib"])
    free_gib = shutil.disk_usage(args.output_root).free / 1024**3
    cleanup_source_audio = (
        args.delete_source_audio and free_gib <= cleanup_trigger_gib
    )
    if cleanup_source_audio:
        for source in sources:
            Path(source["audio_path"]).unlink(missing_ok=True)
    report = {
        "status": "PASS",
        "source_files": len(sources),
        "processed_source_files": len(progress),
        "resumed_source_files": resumed_source_files,
        "accepted_segments": accepted_count,
        "rejected": dict(sorted(rejected.items())),
        "artifact": str(args.output_records),
        "progress_artifact": str(progress_path),
        "diarization_maximum_chunk_seconds": float(
            diar_cfg["maximum_chunk_seconds"]
        ),
        "source_audio_cleanup": (
            "CLEANED" if cleanup_source_audio else "DEFERRED"
        ),
        "cleanup_trigger_gib": cleanup_trigger_gib,
        "free_gib_before_cleanup": round(free_gib, 2),
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    raise SystemExit(main())
