"""Atomic FFmpeg de-clicking and peak limiting for enhanced audio."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import tempfile
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import soundfile as sf
import yaml

LOGGER = logging.getLogger(__name__)


class AudioPostprocessError(RuntimeError):
    """Report a de-clicking or limiting failure."""


def resolve_ffmpeg_binary(ffmpeg_binary: str | None = None) -> str:
    """Select an explicit binary, a local static binary, or the system binary."""
    if ffmpeg_binary and ffmpeg_binary != "auto":
        return ffmpeg_binary
    configured = os.environ.get("UKTTS_FFMPEG")
    if configured:
        return configured
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError):
        return "ffmpeg"


@dataclass(frozen=True)
class DeClickLimiterConfig:
    """Configuration for the final FFmpeg audio stage."""

    declick_window: float = 55.0
    declick_overlap: float = 75.0
    declick_ar_order: float = 2.0
    declick_threshold: float = 4.0
    declick_burst: float = 2.0
    limiter_limit: float = 0.891251
    limiter_attack_ms: float = 5.0
    limiter_release_ms: float = 80.0
    limiter_level: bool = False
    limiter_latency: bool = True

    def __post_init__(self) -> None:
        ranges = {
            "declick_window": (self.declick_window, 10.0, 100.0),
            "declick_overlap": (self.declick_overlap, 50.0, 95.0),
            "declick_ar_order": (self.declick_ar_order, 0.0, 25.0),
            "declick_threshold": (self.declick_threshold, 1.0, 100.0),
            "declick_burst": (self.declick_burst, 0.0, 10.0),
            "limiter_limit": (self.limiter_limit, 0.0625, 1.0),
            "limiter_attack_ms": (self.limiter_attack_ms, 0.1, 80.0),
            "limiter_release_ms": (self.limiter_release_ms, 1.0, 8000.0),
        }
        for name, (value, minimum, maximum) in ranges.items():
            if not minimum <= value <= maximum:
                raise ValueError(
                    f"{name} must be from {minimum:g} to {maximum:g}; got {value:g}"
                )

    @property
    def filter_chain(self) -> str:
        level = str(self.limiter_level).lower()
        latency = str(self.limiter_latency).lower()
        return (
            "aformat=sample_fmts=fltp,"
            f"adeclick=w={self.declick_window:g}:o={self.declick_overlap:g}:"
            f"a={self.declick_ar_order:g}:t={self.declick_threshold:g}:"
            f"b={self.declick_burst:g},"
            f"alimiter=limit={self.limiter_limit:g}:"
            f"attack={self.limiter_attack_ms:g}:"
            f"release={self.limiter_release_ms:g}:"
            f"level={level}:latency={latency}"
        )

    @property
    def digest(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def config_from_mapping(values: Mapping[str, Any]) -> DeClickLimiterConfig:
    """Make a validated configuration and reject unknown values."""
    allowed = set(DeClickLimiterConfig.__dataclass_fields__)
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"Unknown de-click configuration values: {unknown}")
    return DeClickLimiterConfig(**dict(values))


def load_config(path: Path | None) -> DeClickLimiterConfig:
    """Load YAML or JSON configuration, or return the fixed defaults."""
    if path is None:
        return DeClickLimiterConfig()
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise AudioPostprocessError(
            f"Cannot read post-processing config {path}: {error}"
        ) from error
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        raise AudioPostprocessError("The post-processing config must be a mapping.")
    values = payload.get("declick_peak_limit", payload)
    if not isinstance(values, dict):
        raise AudioPostprocessError("declick_peak_limit must be a mapping.")
    try:
        return config_from_mapping(values)
    except (TypeError, ValueError) as error:
        raise AudioPostprocessError(
            f"Invalid post-processing config: {error}"
        ) from error


def apply_overrides(
    config: DeClickLimiterConfig,
    values: Mapping[str, Any],
) -> DeClickLimiterConfig:
    """Apply non-null CLI overrides to a validated configuration."""
    selected = {key: value for key, value in values.items() if value is not None}
    try:
        return replace(config, **selected)
    except (TypeError, ValueError) as error:
        raise AudioPostprocessError(
            f"Invalid post-processing override: {error}"
        ) from error


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_audio(path: Path) -> dict[str, Any]:
    """Read the container properties that the final stage must preserve."""
    try:
        info = sf.info(path)
    except (OSError, RuntimeError) as error:
        raise AudioPostprocessError(f"Cannot inspect audio {path}: {error}") from error
    if info.frames <= 0 or info.samplerate <= 0 or info.channels <= 0:
        raise AudioPostprocessError(f"Audio has invalid stream properties: {path}")
    return {
        "sample_rate": int(info.samplerate),
        "channels": int(info.channels),
        "frames": int(info.frames),
        "duration": float(info.frames / info.samplerate),
        "format": str(info.format),
        "subtype": str(info.subtype),
    }


@lru_cache(maxsize=8)
def validate_ffmpeg(ffmpeg_binary: str) -> str:
    """Require the filters and latency compensation used by the fixed graph."""
    try:
        version = subprocess.run(
            [ffmpeg_binary, "-version"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()[0]
        adeclick = subprocess.run(
            [ffmpeg_binary, "-hide_banner", "-h", "filter=adeclick"],
            check=True,
            capture_output=True,
            text=True,
        )
        alimiter = subprocess.run(
            [ffmpeg_binary, "-hide_banner", "-h", "filter=alimiter"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError, IndexError) as error:
        raise AudioPostprocessError(
            f"FFmpeg is not usable at {ffmpeg_binary!r}: {error}"
        ) from error
    adeclick_help = adeclick.stdout + adeclick.stderr
    alimiter_help = alimiter.stdout + alimiter.stderr
    if "Filter adeclick" not in adeclick_help:
        raise AudioPostprocessError("FFmpeg does not provide the adeclick filter.")
    if "Filter alimiter" not in alimiter_help:
        raise AudioPostprocessError("FFmpeg does not provide the alimiter filter.")
    if "latency" not in alimiter_help:
        raise AudioPostprocessError(
            "FFmpeg alimiter does not support latency compensation. "
            "Use the pinned FFmpeg binary or set --ffmpeg to a compatible binary."
        )
    return version


def process_audio_file(
    source: Path,
    target: Path,
    *,
    config: DeClickLimiterConfig | None = None,
    ffmpeg_binary: str = "auto",
    overwrite: bool = False,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Write one PCM24 WAV atomically and do not change the source file."""
    config = config or DeClickLimiterConfig()
    logger = logger or LOGGER
    source = source.expanduser().resolve(strict=True)
    target = target.expanduser().absolute()
    if target.suffix.casefold() != ".wav":
        raise AudioPostprocessError("The final output must have a .wav suffix.")
    if source == target or (target.exists() and os.path.samefile(source, target)):
        raise AudioPostprocessError("The output path must differ from the source path.")
    if target.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {target}")

    ffmpeg_binary = resolve_ffmpeg_binary(ffmpeg_binary)
    ffmpeg_version = validate_ffmpeg(ffmpeg_binary)
    source_info = inspect_audio(source)
    source_hash = sha256(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.stem}.", suffix=".wav", dir=target.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    command = [
        ffmpeg_binary,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-map",
        "0:a:0",
        "-vn",
        "-af",
        config.filter_chain,
        "-c:a",
        "pcm_s24le",
        str(temporary),
    ]
    logger.info("Start de-click and limit: %s -> %s", source, target)
    try:
        run = subprocess.run(command, check=False, capture_output=True, text=True)
        if run.returncode:
            message = run.stderr.strip() or "FFmpeg returned no error text."
            raise AudioPostprocessError(
                f"FFmpeg failed for {source} with exit {run.returncode}: {message}"
            )
        output_info = inspect_audio(temporary)
        if output_info["sample_rate"] != source_info["sample_rate"]:
            raise AudioPostprocessError(
                "The final stage changed the sample rate: "
                f"{source_info['sample_rate']} -> {output_info['sample_rate']}"
            )
        if output_info["channels"] != source_info["channels"]:
            raise AudioPostprocessError(
                "The final stage changed the channel count: "
                f"{source_info['channels']} -> {output_info['channels']}"
            )
        if (
            output_info["format"] not in {"WAV", "WAVEX"}
            or output_info["subtype"] != "PCM_24"
        ):
            raise AudioPostprocessError(
                "The final output is not WAV-compatible PCM_24: "
                f"{output_info['format']}/{output_info['subtype']}"
            )
        if sha256(source) != source_hash:
            raise AudioPostprocessError(
                f"The source file changed during processing: {source}"
            )
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    except Exception:
        temporary.unlink(missing_ok=True)
        logger.exception("De-click and limit failed: %s", source)
        raise

    output_hash = sha256(target)
    logger.info("Completed de-click and limit: %s", target)
    return {
        "status": "PASS",
        "source": str(source),
        "output": str(target),
        "source_sha256": source_hash,
        "output_sha256": output_hash,
        "source_audio": source_info,
        "output_audio": output_info,
        "filter_chain": config.filter_chain,
        "processing_config": asdict(config),
        "processing_config_hash": config.digest,
        "ffmpeg_version": ffmpeg_version,
        "atomic_write": True,
        "source_preserved": True,
    }
