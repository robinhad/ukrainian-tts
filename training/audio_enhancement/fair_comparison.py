"""Shared audio controls and measurements for a fair enhancement comparison."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

LOUDNORM_JSON = re.compile(r"\{\s*\"input_i\".*?\}", re.DOTALL)
PEAK_CEILING_DBFS = -0.1


def sha256(path: Path) -> str:
    """Calculate the SHA-256 value for one file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fit_sample_count(audio: np.ndarray, sample_count: int) -> np.ndarray:
    """Crop or zero-pad audio to an exact sample count."""
    waveform = np.asarray(audio, dtype=np.float32)
    if waveform.ndim == 1:
        waveform = waveform[:, None]
    if len(waveform) >= sample_count:
        return waveform[:sample_count].copy()
    return np.pad(waveform, ((0, sample_count - len(waveform)), (0, 0)))


def noise_floor_dbfs(audio: np.ndarray, sample_rate: int) -> float:
    """Estimate the noise floor from the quietest 10 percent of short frames."""
    waveform = np.asarray(audio, dtype=np.float64)
    if waveform.ndim == 2:
        waveform = np.mean(waveform, axis=1)
    frame = max(1, round(0.020 * sample_rate))
    if len(waveform) < frame:
        values = [float(np.sqrt(np.mean(np.square(waveform))))]
    else:
        values = [
            float(np.sqrt(np.mean(np.square(waveform[start : start + frame]))))
            for start in range(0, len(waveform) - frame + 1, frame)
        ]
    return float(20.0 * np.log10(max(np.percentile(values, 10), 1e-12)))


def hnr_db(audio: np.ndarray, sample_rate: int) -> float | None:
    """Estimate HNR from the normalized autocorrelation of voiced frames."""
    waveform = np.asarray(audio, dtype=np.float64)
    if waveform.ndim == 2:
        waveform = np.mean(waveform, axis=1)
    frame = max(16, round(0.040 * sample_rate))
    hop = max(8, frame // 2)
    lag_min = max(1, round(sample_rate / 500.0))
    lag_max = min(frame - 2, round(sample_rate / 70.0))
    values = []
    window = np.hanning(frame)
    for start in range(0, max(0, len(waveform) - frame + 1), hop):
        item = waveform[start : start + frame]
        item = (item - np.mean(item)) * window
        energy = float(np.dot(item, item))
        if energy < 1e-8:
            continue
        correlation = np.correlate(item, item, mode="full")[frame - 1 :]
        periodicity = float(np.max(correlation[lag_min : lag_max + 1]) / energy)
        if periodicity < 0.1:
            continue
        periodicity = min(max(periodicity, 1e-6), 1.0 - 1e-6)
        values.append(10.0 * math.log10(periodicity / (1.0 - periodicity)))
    return float(np.median(values)) if values else None


def measure_lufs(audio: np.ndarray, sample_rate: int, ffmpeg: str = "ffmpeg") -> float:
    """Measure integrated loudness with the FFmpeg EBU R128 implementation."""
    with tempfile.TemporaryDirectory(prefix="uktts-fair-loudness-") as temporary:
        path = Path(temporary) / "measure.wav"
        sf.write(path, np.asarray(audio, dtype=np.float32), sample_rate, subtype="FLOAT")
        command = [
            ffmpeg,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "info",
            "-i",
            str(path),
            "-af",
            "loudnorm=I=-23:LRA=7:TP=-1:print_format=json",
            "-f",
            "null",
            "-",
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=True)
    matches = LOUDNORM_JSON.findall(result.stderr)
    if not matches:
        raise RuntimeError("FFmpeg did not return a loudness measurement")
    value = json.loads(matches[-1])["input_i"]
    measured = float(value)
    if not math.isfinite(measured):
        raise RuntimeError(f"invalid integrated loudness: {value}")
    return measured


def match_loudness_and_prevent_clipping(
    audio: np.ndarray,
    sample_rate: int,
    target_lufs: float,
    *,
    ffmpeg: str = "ffmpeg",
    ceiling_dbfs: float = PEAK_CEILING_DBFS,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply linear loudness gain and cap the gain before digital clipping."""
    waveform = np.asarray(audio, dtype=np.float32)
    measured_before = measure_lufs(waveform, sample_rate, ffmpeg)
    requested_gain_db = float(target_lufs - measured_before)
    requested_gain = 10.0 ** (requested_gain_db / 20.0)
    peak_before = float(np.max(np.abs(waveform)))
    ceiling = 10.0 ** (ceiling_dbfs / 20.0)
    maximum_gain = ceiling / max(peak_before, 1e-12)
    applied_gain = min(requested_gain, maximum_gain)
    matched = waveform * np.float32(applied_gain)
    matched = np.clip(matched, -ceiling, ceiling).astype(np.float32)
    measured_after = measure_lufs(matched, sample_rate, ffmpeg)
    return matched, {
        "method": "linear_gain_with_peak_safe_cap",
        "target_lufs": target_lufs,
        "measured_before_lufs": measured_before,
        "measured_after_lufs": measured_after,
        "difference_from_target_lu": measured_after - target_lufs,
        "requested_gain_db": requested_gain_db,
        "applied_gain_db": 20.0 * math.log10(max(applied_gain, 1e-12)),
        "peak_gain_cap_applied": bool(applied_gain < requested_gain),
        "ceiling_dbfs": ceiling_dbfs,
    }


def waveform_metrics(
    audio: np.ndarray,
    sample_rate: int,
    *,
    lufs: float | None = None,
) -> dict[str, Any]:
    """Measure the required objective values for one waveform."""
    waveform = np.asarray(audio, dtype=np.float32)
    if waveform.ndim == 1:
        waveform = waveform[:, None]
    peak = float(np.max(np.abs(waveform))) if waveform.size else 0.0
    return {
        "hnr_db": hnr_db(waveform, sample_rate),
        "noise_floor_dbfs": noise_floor_dbfs(waveform, sample_rate),
        "peak_absolute": peak,
        "peak_dbfs": float(20.0 * np.log10(max(peak, 1e-12))),
        "integrated_loudness_lufs": lufs,
        "duration_seconds": float(len(waveform) / sample_rate),
        "sample_count": int(len(waveform)),
        "sample_rate": int(sample_rate),
        "channel_count": int(waveform.shape[1]),
        "finite": bool(np.isfinite(waveform).all()),
    }


def atomic_write_pcm24(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    """Write a physical PCM 24-bit WAV through a temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".wav", dir=path.parent
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        sf.write(temporary, np.asarray(audio, dtype=np.float32), sample_rate, subtype="PCM_24")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"output is not a physical file: {path}")
