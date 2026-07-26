#!/usr/bin/env python3
"""Diarize unlabeled audio and accept high-confidence Ukrainian transcripts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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
    parser.add_argument("--limit", type=int)
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--delete-source-audio", action="store_true")
    args = parser.parse_args()
    registry = load_registry(args.registry)
    diar_cfg = registry["models"]["diarizer"]
    asr_cfg = registry["models"]["asr"]
    diar_model, asr_model = load_models(registry, args.model_cache)
    sources = [
        json.loads(line)
        for path in args.records
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.limit is not None:
        sources = sources[: args.limit]
    accepted = []
    rejected: dict[str, int] = {}
    for source in sources:
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
        predicted = diar_model.diarize(
            audio=[mono], batch_size=1, sample_rate=sample_rate
        )[0]
        parsed = [parse_segment(value) for value in predicted]
        single = non_overlapping_segments(parsed)
        for start, end, speaker in single:
            for part_start, part_end in low_energy_split(
                mono, sample_rate, start, end
            ):
                duration = part_end - part_start
                if not 2.0 <= duration <= 12.0:
                    rejected["duration"] = rejected.get("duration", 0) + 1
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
                    rejected["language"] = rejected.get("language", 0) + 1
                    target.unlink(missing_ok=True)
                    continue
                if confidence is None or confidence < float(
                    asr_cfg["minimum_mean_confidence"]
                ):
                    rejected["confidence"] = rejected.get("confidence", 0) + 1
                    target.unlink(missing_ok=True)
                    continue
                if script_ratio < float(asr_cfg["minimum_ukrainian_letter_ratio"]):
                    rejected["script_ratio"] = rejected.get("script_ratio", 0) + 1
                    target.unlink(missing_ok=True)
                    continue
                accepted_language = language or "uk_script_inferred"
                audio_hash = hashlib.sha256(target.read_bytes()).hexdigest()
                accepted.append(
                    {
                        **source,
                        "utterance_id": identifier,
                        "speaker_id": f"{source['source_id']}:{speaker}",
                        "audio_path": str(target.resolve()),
                        "canonical_raw_audio_path": str(target.resolve()),
                        "text_raw": text,
                        "duration_source": duration,
                        "duration": duration,
                        "sample_rate": sample_rate,
                        "source_group": (
                            f"{source['source_group']}:speaker-{speaker}"
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
    args.output_records.parent.mkdir(parents=True, exist_ok=True)
    with args.output_records.open(
        "a" if args.append else "w", encoding="utf-8"
    ) as stream:
        stream.write(
            "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in accepted
            )
        )
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
        "accepted_segments": len(accepted),
        "rejected": dict(sorted(rejected.items())),
        "artifact": str(args.output_records),
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
