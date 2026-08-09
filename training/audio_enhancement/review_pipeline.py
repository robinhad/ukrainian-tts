"""Shared processing for the enhancement-backend listening review."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from training.scripts.prepare_audio import trim_silence


LOUDNESS_JSON = re.compile(r"\{\s*\"input_i\".*?\}", re.DOTALL)


@dataclass(frozen=True)
class ReviewProcessingConfig:
    """Pinned common processing values for all new review candidates."""

    highpass_hz: float = 70.0
    deesser_intensity: float = 0.15
    deesser_max: float = 0.25
    deesser_frequency: float = 0.50
    target_lufs: float = -23.0
    target_lra: float = 7.0
    target_true_peak_db: float = -1.0
    output_sample_rate: int = 24_000
    trim_top_db: float = 40.0
    trim_padding_ms: float = 100.0
    trim_frame_length: int = 1024
    trim_hop_length: int = 256
    compression_applied: bool = False

    @property
    def digest(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @property
    def mastering_filter(self) -> str:
        return (
            f"highpass=f={self.highpass_hz}:p=2:t=q:w=0.707,"
            f"deesser=i={self.deesser_intensity}:m={self.deesser_max}:"
            f"f={self.deesser_frequency}:s=o"
        )


@dataclass(frozen=True)
class SidonDeessOnlyConfig:
    """Pinned processing values for the Sidon and de-essing sample."""

    deesser_intensity: float = 0.15
    deesser_max: float = 0.25
    deesser_frequency: float = 0.50
    output_sample_rate: int = 24_000
    trim_top_db: float = 40.0
    trim_padding_ms: float = 100.0
    trim_frame_length: int = 1024
    trim_hop_length: int = 256
    boundary_trim_applied: bool = True
    sidon_applied: bool = True
    deessing_applied: bool = True
    post_highpass_applied: bool = False
    loudness_normalization_applied: bool = False
    compression_applied: bool = False
    limiting_applied: bool = False

    @property
    def digest(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @property
    def deesser_filter(self) -> str:
        return (
            f"deesser=i={self.deesser_intensity}:m={self.deesser_max}:"
            f"f={self.deesser_frequency}:s=o"
        )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_loudnorm(stderr: str) -> dict[str, float | str]:
    matches = LOUDNESS_JSON.findall(stderr)
    if not matches:
        raise RuntimeError("ffmpeg loudnorm did not emit JSON measurements")
    raw = json.loads(matches[-1])
    parsed: dict[str, float | str] = {}
    for key, value in raw.items():
        try:
            parsed[key] = float(value)
        except (TypeError, ValueError):
            parsed[key] = str(value)
    return parsed


def load_and_trim(
    source: Path,
    config: ReviewProcessingConfig | SidonDeessOnlyConfig,
) -> tuple[np.ndarray, int, dict[str, Any]]:
    decode_method = "libsndfile"
    try:
        audio, sample_rate = sf.read(source, always_2d=True, dtype="float32")
    except (OSError, RuntimeError):
        decode_method = "ffmpeg_pcm_f32le"
        with tempfile.TemporaryDirectory(prefix="uktts-review-decode-") as temporary:
            decoded = Path(temporary) / "decoded.wav"
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
                    "48000",
                    "-c:a",
                    "pcm_f32le",
                    str(decoded),
                ],
                check=True,
            )
            audio, sample_rate = sf.read(
                decoded,
                always_2d=True,
                dtype="float32",
            )
    if not len(audio) or not np.isfinite(audio).all():
        raise RuntimeError(f"invalid source waveform: {source}")
    if audio.shape[1] != 1:
        audio = np.mean(audio, axis=1, keepdims=True)
    trimmed, metadata = trim_silence(
        audio,
        sample_rate,
        top_db=config.trim_top_db,
        padding_ms=config.trim_padding_ms,
        frame_length=config.trim_frame_length,
        hop_length=config.trim_hop_length,
    )
    if not len(trimmed):
        raise RuntimeError(f"boundary trim produced an empty waveform: {source}")
    metadata["decode_method"] = decode_method
    return trimmed[:, 0].copy(), int(sample_rate), metadata


def write_float_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    waveform = np.asarray(audio, dtype=np.float32).reshape(-1)
    if not waveform.size or not np.isfinite(waveform).all() or not np.any(waveform):
        raise RuntimeError("backend produced an invalid waveform")
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, waveform, sample_rate, subtype="FLOAT")


def master_without_compression(
    source: Path,
    target: Path,
    config: ReviewProcessingConfig,
) -> dict[str, Any]:
    """Apply high-pass, de-essing, and two-pass R128 without compression."""
    with tempfile.TemporaryDirectory(prefix="uktts-review-master-") as temporary:
        mastered = Path(temporary) / "mastered-48k.wav"
        subprocess.run(
            [
                "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                "-i", str(source), "-af", config.mastering_filter, "-ac", "1",
                "-ar", "48000", "-c:a", "pcm_f32le", str(mastered),
            ],
            check=True,
        )
        measure_filter = (
            f"loudnorm=I={config.target_lufs}:LRA={config.target_lra}:"
            f"TP={config.target_true_peak_db}:print_format=json"
        )
        first = subprocess.run(
            [
                "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "info",
                "-i", str(mastered), "-af", measure_filter, "-f", "null", "-",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        measured = parse_loudnorm(first.stderr)
        second_filter = (
            f"loudnorm=I={config.target_lufs}:LRA={config.target_lra}:"
            f"TP={config.target_true_peak_db}:"
            f"measured_I={measured['input_i']}:"
            f"measured_LRA={measured['input_lra']}:"
            f"measured_TP={measured['input_tp']}:"
            f"measured_thresh={measured['input_thresh']}:"
            f"offset={measured['target_offset']}:"
            "linear=true:print_format=json"
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        second = subprocess.run(
            [
                "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "info", "-y",
                "-i", str(mastered), "-af", second_filter, "-ac", "1", "-ar",
                str(config.output_sample_rate), "-c:a", "pcm_s16le", str(target),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    return {"first_pass": measured, "second_pass": parse_loudnorm(second.stderr)}


def apply_deessing_only(
    source: Path,
    target: Path,
    config: SidonDeessOnlyConfig,
) -> dict[str, Any]:
    """Apply only de-essing after Sidon and make a 24 kHz PCM WAV file."""
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
            "-af",
            config.deesser_filter,
            "-ac",
            "1",
            "-ar",
            str(config.output_sample_rate),
            "-c:a",
            "pcm_s16le",
            str(target),
        ],
        check=True,
    )
    return {
        "filter": config.deesser_filter,
        "post_highpass_applied": False,
        "loudness_normalization_applied": False,
        "compression_applied": False,
        "limiting_applied": False,
    }


def inspect_wav(path: Path) -> dict[str, Any]:
    audio, sample_rate = sf.read(path, always_2d=True, dtype="float32")
    info = sf.info(path)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    errors = []
    if path.is_symlink():
        errors.append("symbolic link")
    if sample_rate != 24_000:
        errors.append(f"sample_rate={sample_rate}")
    if info.channels != 1 or audio.shape[1] != 1:
        errors.append(f"channels={info.channels}")
    if not audio.size:
        errors.append("empty waveform")
    if audio.size and not np.isfinite(audio).all():
        errors.append("non-finite waveform")
    if peak == 0.0:
        errors.append("all-zero waveform")
    if peak >= 0.999:
        errors.append("digital clipping")
    return {
        "sample_rate": int(sample_rate),
        "channels": int(info.channels),
        "duration": float(info.frames / sample_rate) if sample_rate else 0.0,
        "format": f"{info.format}/{info.subtype}",
        "peak_absolute": peak,
        "errors": errors,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
