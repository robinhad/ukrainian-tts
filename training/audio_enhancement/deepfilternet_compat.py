"""Compatibility support for DeepFilterNet 0.5.6 with torchaudio 2.9.

DeepFilterNet 0.5.6 imports the removed ``torchaudio.backend.common``
module. This module restores only the metadata API that DeepFilterNet uses.
It does not change PyTorch or torchaudio.
"""

from __future__ import annotations

import sys
import types
from typing import NamedTuple

import soundfile as sf
import torchaudio


class AudioMetaData(NamedTuple):
    """The legacy torchaudio metadata fields used by DeepFilterNet."""

    sample_rate: int
    num_frames: int
    num_channels: int
    bits_per_sample: int
    encoding: str


def _soundfile_info(path: str, **_: object) -> AudioMetaData:
    info = sf.info(path)
    subtype = info.subtype or ""
    bits = 0
    for candidate in (8, 16, 24, 32, 64):
        if str(candidate) in subtype:
            bits = candidate
            break
    return AudioMetaData(
        sample_rate=int(info.samplerate),
        num_frames=int(info.frames),
        num_channels=int(info.channels),
        bits_per_sample=bits,
        encoding=subtype,
    )


def install() -> None:
    """Install the narrow legacy metadata shim before importing ``df.enhance``."""
    backend = sys.modules.setdefault("torchaudio.backend", types.ModuleType("torchaudio.backend"))
    common = sys.modules.setdefault(
        "torchaudio.backend.common",
        types.ModuleType("torchaudio.backend.common"),
    )
    common.AudioMetaData = AudioMetaData
    backend.common = common
    if not hasattr(torchaudio, "info"):
        torchaudio.info = _soundfile_info  # type: ignore[attr-defined]

