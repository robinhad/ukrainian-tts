"""Synthesize a raw 24 kHz mono WAV with the training frontend."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from training.frontend.phonemize import UkrainianPhonemizer


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_speaker_embedding(path: Path, speaker: str) -> np.ndarray:
    from kaldiio import load_ark

    vectors = {
        key: np.asarray(value, dtype=np.float32).squeeze()
        for key, value in load_ark(str(path))
    }
    if speaker not in vectors:
        raise RuntimeError(
            f"speaker {speaker!r} is not in {path}; available={sorted(vectors)}"
        )
    vector = vectors[speaker]
    if vector.ndim != 1 or not np.isfinite(vector).all() or not np.any(vector):
        raise RuntimeError(f"speaker {speaker!r} has an invalid embedding")
    return vector


def synthesize(
    text: str,
    output: Path,
    config: Path,
    checkpoint: Path,
    device: str,
    speaker_embedding: np.ndarray | None = None,
    speaker: str | None = None,
) -> dict:
    from espnet2.bin.tts_inference import Text2Speech

    frontend = UkrainianPhonemizer()
    sanitized, tokens = frontend.phonemize(text)
    model = Text2Speech(train_config=config, model_file=checkpoint, device=device)
    started = time.monotonic()
    result = model(sanitized, spembs=speaker_embedding)
    elapsed = time.monotonic() - started
    waveform = result["wav"].view(-1).detach().cpu().numpy().astype(np.float32)
    sample_rate = int(model.fs)
    if sample_rate != 24000:
        raise RuntimeError(f"checkpoint sample rate is {sample_rate}, expected 24000")
    if waveform.size == 0 or not np.isfinite(waveform).all() or not np.any(waveform):
        raise RuntimeError("model returned an empty, non-finite or all-zero waveform")
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output, waveform, sample_rate, subtype="PCM_16")
    duration = waveform.size / sample_rate
    metadata = {
        "text_raw": text,
        "text_sanitized": sanitized,
        "espeak_phonemes": tokens,
        "frontend_config_hash": frontend.config.digest,
        "espeak_version": frontend.config.espeak_version,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": file_hash(checkpoint),
        "config": str(config.resolve()),
        "config_sha256": file_hash(config),
        "wav": str(output.resolve()),
        "sample_rate": sample_rate,
        "channels": 1,
        "duration": duration,
        "generation_seconds": elapsed,
        "real_time_factor": elapsed / duration,
        "peak_absolute": float(np.max(np.abs(waveform))),
        "warnings": ["possible_clipping"] if np.max(np.abs(waveform)) >= 0.999 else [],
        "speaker": speaker,
        "speaker_embedding_dimension": (
            int(speaker_embedding.size) if speaker_embedding is not None else None
        ),
    }
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--speaker-embedding-ark", type=Path)
    parser.add_argument("--speaker")
    args = parser.parse_args()
    if bool(args.speaker_embedding_ark) != bool(args.speaker):
        parser.error("--speaker-embedding-ark and --speaker must be used together")
    embedding = (
        load_speaker_embedding(args.speaker_embedding_ark, args.speaker)
        if args.speaker_embedding_ark
        else None
    )
    print(
        json.dumps(
            synthesize(
                args.text,
                args.output,
                args.config,
                args.checkpoint,
                args.device,
                speaker_embedding=embedding,
                speaker=args.speaker,
            ),
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
