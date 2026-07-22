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


def synthesize(text: str, output: Path, config: Path, checkpoint: Path, device: str) -> dict:
    from espnet2.bin.tts_inference import Text2Speech

    frontend = UkrainianPhonemizer()
    sanitized, tokens = frontend.phonemize(text)
    model = Text2Speech(train_config=config, model_file=checkpoint, device=device)
    started = time.monotonic()
    result = model(sanitized)
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
    args = parser.parse_args()
    print(json.dumps(synthesize(args.text, args.output, args.config, args.checkpoint, args.device), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
