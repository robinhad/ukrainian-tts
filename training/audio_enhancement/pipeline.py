"""DeepFilterNet3 and conservative two-pass EBU R128 processing."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF

from training.audio_enhancement.deepfilternet_compat import install

install()

from df.enhance import enhance, init_df  # noqa: E402


MODEL_SAMPLE_RATE = 48_000
OUTPUT_SAMPLE_RATE = 24_000
LOUDNESS_JSON = re.compile(r"\{\s*\"input_i\".*?\}", re.DOTALL)


@dataclass(frozen=True)
class EnhancementConfig:
    """Pinned values for the enhanced-data experiment."""

    deepfilter_model: str = "DeepFilterNet3"
    attenuation_limit_db: float = 18.0
    highpass_hz: float = 70.0
    deesser_intensity: float = 0.15
    deesser_max: float = 0.25
    deesser_frequency: float = 0.50
    compressor_threshold: float = 0.1258925412
    compressor_ratio: float = 1.5
    compressor_attack_ms: float = 20.0
    compressor_release_ms: float = 250.0
    target_lufs: float = -23.0
    target_lra: float = 7.0
    target_true_peak_db: float = -1.0
    output_sample_rate: int = OUTPUT_SAMPLE_RATE

    @property
    def digest(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @property
    def mastering_filter(self) -> str:
        return (
            f"highpass=f={self.highpass_hz}:p=2:t=q:w=0.707,"
            f"deesser=i={self.deesser_intensity}:m={self.deesser_max}:"
            f"f={self.deesser_frequency}:s=o,"
            f"acompressor=threshold={self.compressor_threshold}:"
            f"ratio={self.compressor_ratio}:attack={self.compressor_attack_ms}:"
            f"release={self.compressor_release_ms}:makeup=1:knee=2.828427:"
            "detection=rms"
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
    result: dict[str, float | str] = {}
    for key, value in raw.items():
        try:
            result[key] = float(value)
        except (TypeError, ValueError):
            result[key] = str(value)
    return result


class EnhancedAudioProcessor:
    """Load DeepFilterNet3 once and process many files on one CUDA device."""

    def __init__(self, config: EnhancementConfig, model_cache: Path) -> None:
        self.config = config
        model_cache.mkdir(parents=True, exist_ok=True)
        os.environ["XDG_CACHE_HOME"] = str(model_cache.resolve())
        self.model, self.df_state, _ = init_df(
            config.deepfilter_model,
            post_filter=False,
            log_file=None,
            log_level="WARNING",
        )

    def denoise(self, source: Path, output_48k: Path) -> None:
        audio, sample_rate = sf.read(source, always_2d=True, dtype="float32")
        if audio.shape[1] != 1:
            audio = np.mean(audio, axis=1, keepdims=True)
        tensor = torch.from_numpy(audio.T.copy())
        if sample_rate != MODEL_SAMPLE_RATE:
            tensor = AF.resample(tensor, sample_rate, MODEL_SAMPLE_RATE)
        result = enhance(
            self.model,
            self.df_state,
            tensor,
            pad=True,
            atten_lim_db=self.config.attenuation_limit_db,
        )
        sf.write(output_48k, result.squeeze(0).numpy(), MODEL_SAMPLE_RATE, subtype="FLOAT")

    def master_and_normalize(self, source_48k: Path, target: Path) -> dict[str, Any]:
        cfg = self.config
        mastered = source_48k.with_name("mastered-48k.wav")
        subprocess.run(
            [
                "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                "-i", str(source_48k), "-af", cfg.mastering_filter, "-ac", "1",
                "-ar", str(MODEL_SAMPLE_RATE), "-c:a", "pcm_f32le", str(mastered),
            ],
            check=True,
        )
        loudness_filter = (
            f"loudnorm=I={cfg.target_lufs}:LRA={cfg.target_lra}:"
            f"TP={cfg.target_true_peak_db}:print_format=json"
        )
        first = subprocess.run(
            [
                "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "info", "-i",
                str(mastered), "-af", loudness_filter, "-f", "null", "-",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        measured = parse_loudnorm(first.stderr)
        second_filter = (
            f"loudnorm=I={cfg.target_lufs}:LRA={cfg.target_lra}:"
            f"TP={cfg.target_true_peak_db}:"
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
                str(cfg.output_sample_rate), "-c:a", "pcm_s16le", str(target),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        output = parse_loudnorm(second.stderr)
        return {"first_pass": measured, "second_pass": output}

    def process(self, source: Path, target: Path) -> dict[str, Any]:
        with tempfile.TemporaryDirectory(prefix="uktts-dfn3-") as temporary:
            denoised = Path(temporary) / "denoised-48k.wav"
            self.denoise(source, denoised)
            loudness = self.master_and_normalize(denoised, target)
        audio, sample_rate = sf.read(target, always_2d=True, dtype="float32")
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if sample_rate != self.config.output_sample_rate:
            raise RuntimeError(f"unexpected output sample rate: {sample_rate}")
        if audio.shape[1] != 1 or not len(audio) or not np.isfinite(audio).all():
            raise RuntimeError("enhanced output failed waveform validation")
        return {
            "audio_sha256": sha256(target),
            "channels": int(audio.shape[1]),
            "duration": len(audio) / sample_rate,
            "enhancement_config_hash": self.config.digest,
            "format": "WAV/PCM_16",
            "loudness": loudness,
            "peak_amplitude": peak,
            "sample_rate": sample_rate,
        }
